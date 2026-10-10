"""Fetch public benchmark files; raw records remain excluded from Git."""

from pathlib import Path
import shutil
import urllib.request
import zipfile


def download(url, path):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    temporary.replace(path)


if __name__ == "__main__":
    data = Path("data")
    data.mkdir(exist_ok=True)
    download("https://openml.org/data/v1/download/1673544/creditcard.arff", data / "creditcard.arff")
    archive = data / "credit-default.zip"
    download(
        "https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip", archive
    )
    target = data / "credit-default"
    target.mkdir(exist_ok=True)
    # Extract exactly the known data file instead of trusting archive paths.
    filename = "default of credit card clients.xls"
    with (
        zipfile.ZipFile(archive) as zipped,
        zipped.open(filename) as source,
        (target / filename).open("wb") as output,
    ):
        shutil.copyfileobj(source, output)
    print("Downloaded public benchmarks into data/. Cite sources listed in README.")
