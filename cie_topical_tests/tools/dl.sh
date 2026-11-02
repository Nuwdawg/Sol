#!/bin/bash
# usage: dl.sh NAME OUTDIR
n="$1"; out="$2/$n.pdf"
[ -s "$out" ] && exit 0
code=$(curl -sS --retry 3 --retry-delay 2 -o "$out.part" -w "%{http_code}" "https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/$n.pdf")
if [ "$code" = "200" ] && head -c 5 "$out.part" | grep -q '%PDF'; then mv "$out.part" "$out"; else rm -f "$out.part"; echo "MISS $n $code" >> "$2/../work/missing.txt"; fi
