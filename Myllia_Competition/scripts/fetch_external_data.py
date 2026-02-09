import os, sys, json, hashlib, gzip, shutil, zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[1]
EXT  = ROOT / "external"
EXT.mkdir(exist_ok=True)

MANIFEST = {
    "genept": {
        "url": "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1",
        "dst": "external/genept/GenePT_emebdding_v2.zip",
        "extract_to": "external/genept",
    },
    "hgnc": {
        "url": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        "dst": "external/hgnc/hgnc_complete_set.txt",
    },
    "string_aliases": {
        "url": "https://stringdb-downloads.org/download/protein.aliases.v12.0/9606.protein.aliases.v12.0.txt.gz",
        "dst": "external/string/9606.protein.aliases.v12.0.txt.gz",
    },
    "string_links": {
        "url": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
        "dst": "external/string/9606.protein.links.v12.0.txt.gz",
    },
    "reactome_uniprot2reactome": {
        "url": "https://reactome.org/download/current/UniProt2Reactome.txt",
        "dst": "external/reactome/UniProt2Reactome.txt",
    },
    "goa_human_gaf": {
        "url": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
        "dst": "external/go/goa_human.gaf.gz",
    },
    # Full scPerturb h5ad collection (very large)
    #"scperturb_zenodo_rna_protein_h5ad": {
    #    "url": "https://zenodo.org/records/13350497/files/scPerturb_h5ad_files.zip?download=1",
    #    "dst": "external/scperturb/scPerturb_h5ad_files.zip",
    #    "extract_to": "external/scperturb",
    #    "optional": True,
    #},
}
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

def download(url: str, dst: Path, force: bool = False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] {dst}")
        return

    print(f"[dl] {url}\n -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept": "*/*",
    }

    req = Request(url, headers=headers)

    try:
        with urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        tmp.rename(dst)
    except HTTPError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"HTTP {e.code} downloading {url}") from e
    except URLError as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Network error downloading {url}: {e}") from e


def extract_zip(zip_path: Path, to_dir: Path):
    print(f"[unzip] {zip_path} -> {to_dir}")
    to_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(to_dir)

def gunzip(gz_path: Path, out_path: Path, force: bool = False):
    if out_path.exists() and not force:
        return
    print(f"[gunzip] {gz_path} -> {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Download Tier 0/1 (recommended)")
    p.add_argument("--optional", action="store_true", help="Also download large optional datasets")
    p.add_argument("--force", action="store_true", help="Re-download even if files exist")
    p.add_argument("--clean", action="store_true", help="Delete external/ before installing")
    args = p.parse_args()

    if args.clean:
        if EXT.exists():
            print("[clean] removing external/")
            shutil.rmtree(EXT)
        EXT.mkdir(exist_ok=True)

    # Tier 0/1 selection
    keys = [
        "genept", "hgnc",
        "string_aliases", "string_links",
        "reactome_uniprot2reactome",
        "goa_human_gaf",
    ]
    if args.optional:
        keys += ["scperturb_zenodo_rna_protein_h5ad"]

    if not args.all and not args.optional:
        # default behavior: act like --all
        args.all = True

    for k in keys:
        item = MANIFEST[k]
        dst = ROOT / item["dst"]
        download(item["url"], dst, force=args.force)

        if "extract_to" in item and dst.suffix.lower() == ".zip":
            extract_zip(dst, ROOT / item["extract_to"])

    for k in ["string_aliases", "string_links"]:
        gz = ROOT / MANIFEST[k]["dst"]
        out = gz.with_suffix("")  # remove .gz
        gunzip(gz, out, force=args.force)

    print("\nDone")

if __name__ == "__main__":
    main()