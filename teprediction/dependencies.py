"""Fetch the two verified official MoLE runtime files into a separate local directory."""
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = "a4f9fa8a6640598e953464f39f021a641cd9146c"
FILES = {
    "models/MoLE_DLinear.py": "fe5dad01208b0f775923cde1beb5edd1fd80032fd6fb06a0b316300fb6f93c90",
    "utils/headdropout.py": "ff00049e2a30b24d4f47e9b7378f071991919cc3e3e70debbd0ac1841ace6240",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("model_sources/MoLE"))
    args = parser.parse_args()
    destination = args.destination.resolve()
    package = Path(__file__).resolve().parent
    if destination == package or package in destination.parents:
        parser.error("Keep externally obtained MoLE outside the installed package.")
    payloads = {}
    for name, expected in FILES.items():
        url = f"https://raw.githubusercontent.com/RogerNi/MoLE/{COMMIT}/{name}"
        with urlopen(url, timeout=60) as response:
            data = response.read().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Upstream hash mismatch: {name}")
        target = destination/name
        if target.exists() and target.read_bytes().replace(b"\r\n", b"\n") != data:
            raise FileExistsError(f"Refusing to overwrite a different file: {target}")
        payloads[target] = data
    for target, data in payloads.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"Verified MoLE files saved to {destination}")
    print("No upstream license was located. This fetch does not grant redistribution rights.")


if __name__ == "__main__":
    main()
