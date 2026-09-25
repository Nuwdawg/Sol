"""Package the technical report for compilation on a free Kaggle CPU kernel.

This machine has no TeX installation, and the user's bandwidth is not spent on one.
The paper sources travel as one tarball in a private dataset, named `.payload` because
Kaggle auto-extracts recognised archives (see package_exp139_code.py). The kernel installs
TeX Live, checks the tarball's SHA256, runs pdflatex, bibtex, pdflatex twice and writes
main.pdf.

Usage:
    python scripts/package_paper.py
    cd kaggle/paper_src_dataset && kaggle datasets version -p . -m "..." -r skip   # or create
    (wait for the dataset to show ready)
    cd kaggle/paper_build_cpu && kaggle kernels push -p .
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DATASET_ID = "cosmicwitness/sol-paper-src"
KERNEL_ID = "cosmicwitness/sol-paper-build-cpu"
PAYLOAD = "sol_paper.payload"
SOURCES = ["main.tex", "refs.bib", "generated/numbers.tex"]
FIGURES = ["fig_qualitative.pdf", "fig_failures.pdf", "fig_host_metrics.pdf", "fig_curve.pdf",
           "fig_size.pdf", "thumb_input.png", "thumb_output.png"]

KERNEL = r'''import hashlib, subprocess, tarfile, shutil, re
from pathlib import Path
payload = next(Path("/kaggle/input").rglob("__PAYLOAD__"))
digest = hashlib.sha256(payload.read_bytes()).hexdigest()
assert digest == "__SHA256__", f"stale dataset version: {digest}"
work = Path("/tmp/paper"); work.mkdir(parents=True, exist_ok=True)
with tarfile.open(payload, "r:gz") as tar:
    tar.extractall(work)
print("PAYLOAD_OK", digest, flush=True)
subprocess.run("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
               "texlive-latex-base texlive-latex-recommended texlive-latex-extra "
               "texlive-fonts-recommended texlive-fonts-extra texlive-pictures texlive-publishers "
               "texlive-science texlive-bibtex-extra > /tmp/apt.log 2>&1",
               shell=True, check=True)
print("TEX_READY", flush=True)
def run(*command):
    # TeX writes console output in mixed encodings; strict UTF-8 decoding crashed build v4.
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, errors="replace")
    print("RUN", " ".join(command), "->", result.returncode, flush=True)
    return result
run("pdflatex", "-interaction=nonstopmode", "main.tex")
run("bibtex", "main")
run("pdflatex", "-interaction=nonstopmode", "main.tex")
last = run("pdflatex", "-interaction=nonstopmode", "main.tex")
log = (work / "main.log").read_text(errors="replace")
pages = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
warnings = [line for line in log.splitlines()
            if re.search(r"(Overfull|Underfull \\hbox \(badness 10000|undefined|Error|Missing)", line)]
print("PAGES", pages.group(1) if pages else "none", flush=True)
print("WARNINGS", len(warnings), flush=True)
for line in warnings[:40]:
    print("  ", line, flush=True)
out = Path("/kaggle/working")
for name in ("main.pdf", "main.log", "main.blg"):
    if (work / name).exists():
        shutil.copy(work / name, out / name)
print("KERNEL_COMPLETE", flush=True)
'''


def build_payload() -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for name in SOURCES + [f"figures/{f}" for f in FIGURES]:
            data = (PAPER / name).read_bytes()
            info = tarfile.TarInfo(name=name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
        stream.write(archive.getvalue())
    return compressed.getvalue()


def main() -> None:
    missing = [n for n in SOURCES + [f"figures/{f}" for f in FIGURES] if not (PAPER / n).is_file()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    payload = build_payload()
    digest = hashlib.sha256(payload).hexdigest()
    dataset = ROOT / "kaggle" / "paper_src_dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / PAYLOAD).write_bytes(payload)
    (dataset / "dataset-metadata.json").write_text(json.dumps(
        dict(title="sol paper src", id=DATASET_ID, licenses=[dict(name="CC0-1.0")]), indent=2))
    kernel_dir = ROOT / "kaggle" / "paper_build_cpu"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    kernel = KERNEL.replace("__PAYLOAD__", PAYLOAD).replace("__SHA256__", digest)
    (kernel_dir / "kernel.py").write_text(kernel, encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(dict(
        id=KERNEL_ID, title="sol paper build cpu", code_file="kernel.py", language="python",
        kernel_type="script", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=True,
        dataset_sources=[DATASET_ID], competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
    print(json.dumps(dict(payload_bytes=len(payload), sha256=digest, files=len(SOURCES) + len(FIGURES)), indent=2))


if __name__ == "__main__":
    main()
