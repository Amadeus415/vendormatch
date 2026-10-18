import pytest

from vendormatch.normalize import initials, name_variants, normalize, normalize_street, normalize_zip


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Apple Inc.", "apple"),
        ("APPLE INC", "apple"),
        ("Amazon.com, Inc.", "amazon"),
        ("AMAZON COM INC", "amazon"),
        ("ELI LILLY & Co", "eli lilly"),
        ("Eli Lilly and Company", "eli lilly"),
        ("INTERNATIONAL BUSINESS MACHINES CORP", "intl business machines"),
        ("Intl Business Machines Corporation", "intl business machines"),
        ("The Hackett Group, Inc.", "hackett grp"),
        ("Nestlé S.A.", "nestle"),
        ("Acme L.L.C.", "acme"),
        ("REPUBLIC BANCORP INC /KY/", "republic bancorp"),
        ("SOUTHERN COPPER CORP/", "southern copper"),
        ("AC/DC", "ac dc"),
    ],
)
def test_normalize_legal_forms_case_and_punctuation(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("SQ *BLUE BOTTLE COF", "blue bottle cof"),
        ("TST* JOES PIZZA #0423", "joes pizza"),
        ("PAYPAL *SPOTIFY", "spotify"),
        ("AMZN MKTP US*2K4TW1", "amzn mktp us"),
        ("WALMART STORE 1234", "walmart"),
        ("Remit To: Cisco Systems Inc Attn: Accounts Receivable", "cisco sys"),
        ("OPEN TEXT CORP C/O Lockbox Services", "open text"),
    ],
)
def test_normalize_strips_card_and_ap_wrapping(raw, expected):
    assert normalize(raw) == expected


def test_dba_and_fka_give_both_names():
    assert name_variants("Acme Holdings LLC dba Roadrunner Supply") == ["acme hldg", "roadrunner supply"]
    assert name_variants("Meta Platforms Inc fka Facebook Inc") == ["meta platforms", "facebook"]


def test_single_letters_are_joined():
    assert normalize("I B M") == "ibm"


def test_empty_input():
    assert normalize("") == ""
    assert name_variants(None) == [""]


def test_initials():
    assert initials("intl business machines") == "ibm"
    assert initials("advanced micro devices") == "amd"
    assert initials("apple") == ""  # one strong token is not an acronym source


def test_address_helpers():
    assert normalize_street("100 Main Street, Suite 200") == "100 main st ste 200"
    assert normalize_zip("94043-1351") == "94043"
    assert normalize_zip("123") == ""


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("UFP TECHNOLOGIES INCOR", "ufp tech"),  # cut off by the 22-character card limit
        ("US BANCORP Corl", "us bancorp"),  # one typo away from Corp
        ("ONEOK, Nic.", "oneok"),  # transposed Inc
        ("Hon Inds Inc", "hon ind"),  # 'ind' is a known word, not a misspelled 'inc'
        ("Acme Lab", "acme lab"),
    ],
)
def test_truncated_and_mistyped_suffixes(raw, expected):
    assert normalize(raw) == expected
