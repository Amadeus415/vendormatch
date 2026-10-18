"""vendormatch: entity resolution for vendor and company records.

    from vendormatch import dedupe
    result = dedupe([{"name": "SQ *BLUE BOTTLE COF"}, {"name": "Blue Bottle Coffee, Inc."}])
    result.clusters  # one cluster id per record
"""

from .normalize import name_variants, normalize
from .pipeline import Result, dedupe

__all__ = ["Result", "dedupe", "name_variants", "normalize"]
