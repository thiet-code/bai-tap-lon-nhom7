"""Tải dataset MovieLens 100K từ GroupLens."""
import io
import os
import sys
import zipfile
import urllib.request

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEST = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def main() -> None:
    os.makedirs(DEST, exist_ok=True)
    marker = os.path.join(DEST, "ml-100k", "u.data")
    if os.path.exists(marker):
        print(f"Da co dataset tai: {marker}")
        return

    print(f"Dang tai: {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    print(f"Tai xong: {len(data)/1024:.1f} KB")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(DEST)
    print(f"Da giai nen vao: {DEST}")

    if os.path.exists(marker):
        print("Dataset san sang.")
    else:
        print("LOI: khong thay file u.data sau khi giai nen.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
