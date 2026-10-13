"""Download lt_walks_v1.zip (Causal Chamber, CC BY 4.0) and check its published MD5.

    python scripts/fetch_data.py data/

Source: https://github.com/juangamella/causal-chamber/tree/main/datasets/lt_walks_v1
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ibx.data import ZIP_MD5, ZIP_URL  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
out.mkdir(parents=True, exist_ok=True)
target = out / "lt_walks_v1.zip"
if not target.exists():
    print(f"downloading {ZIP_URL}")
    urllib.request.urlretrieve(ZIP_URL, target)
md5 = hashlib.md5(target.read_bytes()).hexdigest()
if md5 != ZIP_MD5:
    sys.exit(f"MD5 mismatch: {md5} != {ZIP_MD5}")
print(f"ok: {target} ({target.stat().st_size} bytes, md5 {md5})")
