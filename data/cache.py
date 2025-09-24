from functools import lru_cache

import numpy as np
from diskcache import FanoutCache  # type: ignore[import]

from config import CACHE_PATH
from data.ct_scan import CTScan

cache = FanoutCache(directory=CACHE_PATH, shards=64)


@lru_cache(maxsize=1)
def get_ct_scan(series_uid: str) -> CTScan:
    return CTScan(series_uid=series_uid)


@cache.memoize(typed=True)
def get_ct_raw_candidate(
    series_uid: str, center_xyz: tuple[int, int, int], width_irc: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    ct = get_ct_scan(series_uid)
    ct_chunk, center_irc = ct.get_raw_candidate(candidate_xyz=center_xyz, width_irc=width_irc)
    return ct_chunk, center_irc
