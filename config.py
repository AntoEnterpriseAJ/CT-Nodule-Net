from pathlib import Path

DATA_PATH = Path("/mnt/e/dev/data/luna")
CACHE_PATH = DATA_PATH / "cache"
ANNOTATIONS_PATH = DATA_PATH / "annotations.csv"
CANDIDATES_PATH = DATA_PATH / "candidates.csv"


MHD_INDEX = {file.stem: file for file in DATA_PATH.rglob("subset*/*.mhd") if file.is_file()}
