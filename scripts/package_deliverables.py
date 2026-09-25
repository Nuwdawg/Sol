"""Assemble the finished report and onboarding book as two private Kaggle datasets.

Each dataset holds the compiled PDF plus everything needed to rebuild it: LaTeX sources,
the bibliography, the figure scripts, the generated number macros and the figures.
Third-party files in paper/ (the hosts' sample report and the Murphy textbook) are never
copied: they are not ours to redistribute.

Usage:
    python scripts/package_deliverables.py --book-pdf <path to the built book.pdf>
    cd kaggle/deliverable_paper_dataset && kaggle datasets create -p . --dir-mode zip
    cd kaggle/deliverable_book_dataset && kaggle datasets create -p . --dir-mode zip
(`kaggle datasets version -p . -m "..." --dir-mode zip` for later versions.) Datasets are
private unless `--public` is passed. After each upload, list the files and compare sizes.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "paper": dict(id="cosmicwitness/sol-filament-paper", title="sol filament paper",
                  source=ROOT / "paper", pdf="main.pdf",
                  files=["main.tex", "refs.bib", "make_figures.py"],
                  folders=["figures", "generated", "data"]),
    "book": dict(id="cosmicwitness/sol-filament-book", title="sol filament book",
                 source=ROOT / "book", pdf=None,
                 files=["main.tex", "preamble.tex", "refs.bib", "make_book_figures.py"],
                 folders=["chapters", "figures", "generated"]),
}


def assemble(kind: str, book_pdf: Path | None) -> dict:
    spec = DATASETS[kind]
    target = ROOT / "kaggle" / f"deliverable_{kind}_dataset"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    pdf = spec["source"] / spec["pdf"] if spec["pdf"] else book_pdf
    if pdf is None or not pdf.exists():
        raise SystemExit(f"{kind}: compiled PDF not found: {pdf}")
    shutil.copy2(pdf, target / ("paper.pdf" if kind == "paper" else "book.pdf"))
    for name in spec["files"]:
        shutil.copy2(spec["source"] / name, target / name)
    for name in spec["folders"]:
        shutil.copytree(spec["source"] / name, target / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "dataset-metadata.json").write_text(json.dumps(dict(
        title=spec["title"], id=spec["id"], licenses=[dict(name="CC0-1.0")]), indent=2),
        encoding="utf-8")
    listing = {str(path.relative_to(target)).replace("\\", "/"): path.stat().st_size
               for path in sorted(target.rglob("*")) if path.is_file()}
    return dict(dataset=spec["id"], files=len(listing), bytes=sum(listing.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book-pdf", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps([assemble("paper", None), assemble("book", args.book_pdf)], indent=2))


if __name__ == "__main__":
    main()
