import hashlib, subprocess, tarfile, shutil, re
from pathlib import Path
payload = next(Path("/kaggle/input").rglob("sol_book.payload"))
digest = hashlib.sha256(payload.read_bytes()).hexdigest()
assert digest == "f76a46c18511c96e0e4408a0a0ff03226ed27e62a6b532a8eeac4467d024e6b4", f"stale dataset version: {digest}"
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
