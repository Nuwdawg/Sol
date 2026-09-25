import hashlib, subprocess, tarfile, shutil, re
from pathlib import Path
payload = next(Path("/kaggle/input").rglob("sol_paper.payload"))
digest = hashlib.sha256(payload.read_bytes()).hexdigest()
assert digest == "b3c5ab3bb9138202a417073f0fa022758e201acacb87e7d60b1ed48da84bfbcb", f"stale dataset version: {digest}"
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
