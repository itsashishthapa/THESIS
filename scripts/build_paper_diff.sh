#!/usr/bin/env bash
# Compare the current manuscript with HEAD and compile a PDF of the changes.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
paper_dir="$(cd -- "$script_dir/../paper" && pwd)"

for tool in git latexdiff latexdiff-vc latexmk; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "Error: required command '$tool' is not installed or is not on PATH." >&2
        exit 1
    fi
done

cd -- "$paper_dir"
mkdir -p build/changes

latexdiff-vc --git --flatten --force --graphics-markup=none \
    --dir=build/changes -r HEAD main.tex
mv -f -- build/changes/main.tex build/changes/main-diff.tex

latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -outdir=build/changes build/changes/main-diff.tex

printf '\nGenerated: %s/build/changes/main-diff.pdf\n' "$paper_dir"
