"""Package the onboarding book (book/) for compilation on a free Kaggle CPU kernel.

Same contract as package_paper.py: the sources travel as one `.payload` tarball in a
private dataset (Kaggle auto-extracts recognised archives), the kernel checks its
SHA256, installs TeX Live, runs pdflatex, bibtex, pdflatex three times, and writes book.pdf.

Usage:
    PYTHONPATH=. python book/make_book_figures.py --annotations ... --images ...
    python scripts/package_book.py
    cd kaggle/book_src_dataset && kaggle datasets create -p .     # first time
    cd kaggle/book_src_dataset && kaggle datasets version -p . -m "..." -r skip
    cd kaggle/book_build_cpu && kaggle kernels push -p .
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book"
DATASET_ID = "cosmicwitness/sol-book-src"
KERNEL_ID = "cosmicwitness/sol-book-build-cpu"
PAYLOAD = "sol_book.payload"

KERNEL = r'''import hashlib, subprocess, tarfile, shutil, re
from pathlib import Path
payload = next(Path("/kaggle/input").rglob("__PAYLOAD__"))
digest = hashlib.sha256(payload.read_bytes()).hexdigest()
assert digest == "__SHA256__", f"stale dataset version: {digest}"
work = Path("/tmp/book"); work.mkdir(parents=True, exist_ok=True)
with tarfile.open(payload, "r:gz") as tar:
    tar.extractall(work)
print("PAYLOAD_OK", digest, flush=True)
subprocess.run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
               "texlive-latex-base texlive-latex-recommended texlive-latex-extra "
               "texlive-fonts-recommended texlive-fonts-extra texlive-pictures texlive-science "
               "texlive-bibtex-extra > /tmp/apt.log 2>&1", shell=True, check=True)
print("TEX_READY", flush=True)
def run(*command):
    # TeX writes console output in mixed encodings; decode leniently.
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, errors="replace")
    print("RUN", " ".join(command), "->", result.returncode, flush=True)
    return result
run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("bibtex", "main")
run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("pdflatex", "-interaction=nonstopmode", "main.tex")   # longtable widths and page refs settle
log = (work / "main.log").read_text(errors="replace")
pages = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
problems = [line for line in log.splitlines()
            if re.search(r"(^!|Overfull \\hbox \((\d{2,}|[5-9])\.|undefined|Missing|Emergency)", line)]
print("PAGES", pages.group(1) if pages else "none", flush=True)
print("PROBLEMS", len(problems), flush=True)
for line in problems[:80]:
    print("  ", line[:220], flush=True)
out = Path("/kaggle/working")
if (work / "main.pdf").exists():
    shutil.copy(work / "main.pdf", out / "book.pdf")
for name in ("main.log", "main.blg"):
    if (work / name).exists():
        shutil.copy(work / name, out / name)
print("KERNEL_COMPLETE", flush=True)
'''


def sources() -> list[Path]:
    files = [BOOK / "main.tex", BOOK / "preamble.tex", BOOK / "refs.bib"]
    files += sorted((BOOK / "chapters").glob("*.tex"))
    files += sorted((BOOK / "generated").glob("*.tex"))
    files += sorted(p for p in (BOOK / "figures").iterdir() if p.suffix in (".pdf", ".png", ".jpg"))
    return files


def build_payload(files: list[Path]) -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(name=path.relative_to(BOOK).as_posix())
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        stream.write(archive.getvalue())
    return compressed.getvalue()


def main() -> None:
    files = sources()
    payload = build_payload(files)
    digest = hashlib.sha256(payload).hexdigest()
    dataset = ROOT / "kaggle" / "book_src_dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / PAYLOAD).write_bytes(payload)
    (dataset / "dataset-metadata.json").write_text(json.dumps(
        dict(title="sol book src", id=DATASET_ID, licenses=[dict(name="CC0-1.0")]), indent=2))
    kernel_dir = ROOT / "kaggle" / "book_build_cpu"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.py").write_text(KERNEL.replace("__PAYLOAD__", PAYLOAD).replace("__SHA256__", digest),
                                          encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title="sol book build cpu", code_file="kernel.py", language="python",
        kernel_type="script", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
        dataset_sources=[DATASET_ID], competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
    print(json.dumps(dict(payload_bytes=len(payload), sha256=digest, files=len(files)), indent=2))


if __name__ == "__main__":
    main()
