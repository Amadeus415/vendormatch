"""Stage 4: a local LLM judge for the pairs the cheap scorer is unsure about.

Runs on Apple Silicon through MLX (`pip install vendormatch[llm]`). No API keys, nothing leaves the laptop.

The answer is forced into a fixed JSON shape: the prompt ends with `{"match":`, so the very next token
is the decision. Reading the probabilities of " true" and " false" at that position gives a score
(useful for PR curves and thresholds) without any parsing. The short reason is then decoded greedily.

Qwen3's thinking mode is off by default (enable_thinking=False in the chat template). With
thinking_budget > 0 the model may think for up to that many tokens before it has to answer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

DEFAULT_MODEL = "mlx-community/Qwen3-14B-4bit"

SYSTEM = """You reconcile vendor records from different finance systems: the vendor master (ERP), \
AP invoices, corporate card feeds, and procurement/CRM. Decide whether two records refer to the same \
company: the same legal entity, or a brand, dba, abbreviation, or former name of it.

Notes:
- Card descriptors are cut to about 22 characters and may start with a payment processor tag \
such as "SQ *", "TST*" or "PAYPAL *".
- A card transaction often shows the city where the charge happened, not the company's address.
- Different companies often have similar names (for example "Summit Bank" and "Summit \
Therapeutics"). Similar is not the same.
- Numbered entities (Fund II vs Fund III, Acquisition Corp IV vs V) are different companies.

Reply with JSON only: {"match": true or false, "reason": "<at most 15 words>"}"""

SOURCE_LABEL = {
    "erp": "vendor master",
    "ap": "AP invoice",
    "card": "card feed",
    "crm": "procurement/CRM",
}
FIELDS = ("name", "street", "city", "state", "zip")


def describe(record: Mapping[str, str | None], label: str) -> str:
    source = SOURCE_LABEL.get((record.get("source") or "").lower(), record.get("source") or "unknown")
    parts = [f'{f} "{record[f]}"' if f == "name" else f"{f} {record[f]}" for f in FIELDS if record.get(f)]
    return f"Record {label} ({source}): " + "; ".join(parts)


def user_message(a: Mapping[str, str | None], b: Mapping[str, str | None]) -> str:
    return describe(a, "A") + "\n" + describe(b, "B")


@dataclass
class Judgment:
    match: bool
    p_match: float
    reason: str
    prefill_tokens: int  # tokens actually computed (the shared system prompt is cached)
    generated_tokens: int
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


class LLMJudge:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        reason: bool = True,
        max_reason_tokens: int = 40,
        thinking_budget: int = 0,
        seed: int = 0,
    ):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache

        self.mx = mx
        self.model_name = model
        self.model, self.tokenizer = load(model)
        self.cache = make_prompt_cache(self.model)
        self.cached: list[int] = []
        self.reason = reason
        self.max_reason_tokens = max_reason_tokens
        self.thinking_budget = thinking_budget
        mx.random.seed(seed)
        enc = lambda s: self.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
        self.true_id, self.false_id = enc(" true")[0], enc(" false")[0]
        self.reason_prefix = enc(', "reason": "')
        self.think_end = enc("</think>")
        self.close_think = enc("\n</think>\n\n")
        self.after_think = enc("\n\n")
        self.answer_prefix = enc('{"match":')
        self.eos = set(self.tokenizer.eos_token_ids)

    # -- low-level: a prefix-cached forward pass ---------------------------------------------------
    def _feed(self, ids: Sequence[int]):
        """Run ids through the model on top of the cache; return logits for the last position."""
        logits = None
        for start in range(0, len(ids), 512):
            chunk = list(ids[start : start + 512])
            logits = self.model(self.mx.array([chunk]), cache=self.cache)[0, -1]
            self.cached.extend(chunk)
        self.mx.eval(logits)
        return logits

    def _prefill(self, ids: list[int]):
        """Reuse the longest cached prefix (the system prompt), compute only the rest."""
        from mlx_lm.models.cache import trim_prompt_cache

        common = 0
        for x, y in zip(self.cached, ids, strict=False):
            if x != y:
                break
            common += 1
        common = min(common, len(ids) - 1)  # always recompute at least one token to get logits
        trim_prompt_cache(self.cache, len(self.cached) - common)
        del self.cached[common:]
        return self._feed(ids[common:]), len(ids) - common

    def _sample(self, logits, temp: float) -> int:
        if temp <= 0:
            return int(self.mx.argmax(logits).item())
        return int(self.mx.random.categorical(logits / temp).item())

    # -- public ------------------------------------------------------------------------------------
    def prompt_ids(self, a: Mapping, b: Mapping) -> list[int]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(a, b)}]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.thinking_budget > 0
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def warm(self) -> None:
        """Cache the system prompt so the first timed call is not special."""
        self._prefill(self.prompt_ids({"name": "x"}, {"name": "y"}))

    def judge(self, a: Mapping, b: Mapping) -> Judgment:
        t0 = time.perf_counter()
        ids = self.prompt_ids(a, b)
        generated = 0
        if self.thinking_budget > 0:
            logits, prefilled = self._prefill(ids)
            for _ in range(self.thinking_budget):
                tok = self._sample(logits, temp=0.6)
                generated += 1
                logits = self._feed([tok])
                if self.cached[-len(self.think_end) :] == self.think_end or tok in self.eos:
                    logits = self._feed(self.after_think + self.answer_prefix)
                    break
            else:  # budget spent: close the thought and force an answer
                logits = self._feed(self.close_think + self.answer_prefix)
        else:
            logits, prefilled = self._prefill(ids + self.answer_prefix)
        pair = self.mx.softmax(logits[self.mx.array([self.true_id, self.false_id])].astype(self.mx.float32))
        p_match = float(pair[0].item())
        match = p_match >= 0.5
        reason = ""
        if self.reason:
            logits = self._feed([self.true_id if match else self.false_id, *self.reason_prefix])
            out: list[int] = []
            for _ in range(self.max_reason_tokens):
                tok = self._sample(logits, temp=0)
                if tok in self.eos:
                    break
                out.append(tok)
                generated += 1
                if '"' in self.tokenizer.decode([tok]):
                    break
                logits = self._feed([tok])
            reason = self.tokenizer.decode(out).split('"')[0].strip()
        return Judgment(match, p_match, reason, prefilled, generated, time.perf_counter() - t0)
