from functools import lru_cache

from data.ct_scan import CTScan


@lru_cache(maxsize=8)
def get_ct_scan(series_uid: str) -> CTScan:
    return CTScan(series_uid=series_uid)
