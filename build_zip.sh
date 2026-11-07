#!/usr/bin/env bash
# Build the single submission ZIP required by the challenge statement.
#
#   ./build_zip.sh
#
# The ZIP root holds the four required items (submission.py, requirements.txt,
# README.md, results_note.pdf); the rest of the tree is the code that produced
# every number in the note.  There are no trained weights to include.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="$ROOT/dist/stage"
OUT="$ROOT/dist/spinn_poisson_submission.zip"

rm -rf "$STAGE" "$OUT"
mkdir -p "$STAGE"

# copy the repository, minus VCS, build products and caches
tar -C "$ROOT" \
    --exclude=.git \
    --exclude=dist \
    --exclude=.venv \
    --exclude=__pycache__ \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    -cf - . | tar -C "$STAGE" -xf -

# required top-level deliverables
for f in submission.py requirements.txt README.md results_note.pdf; do
    if [[ ! -f "$STAGE/$f" ]]; then
        echo "ERROR: missing required deliverable: $f" >&2
        exit 1
    fi
done

# sanity: the note really is 2-3 pages
if command -v pdfinfo >/dev/null 2>&1; then
    pages="$(pdfinfo "$STAGE/results_note.pdf" | awk '/^Pages:/ {print $2}')"
    echo "results note: ${pages} page(s)"
    if (( pages < 2 || pages > 3 )); then
        echo "ERROR: results note must be 2-3 pages, found ${pages}" >&2
        exit 1
    fi
fi

(cd "$STAGE" && zip -q -r "$OUT" .)
rm -rf "$STAGE"

echo "built: $OUT ($(du -h "$OUT" | cut -f1))"
unzip -l "$OUT" | sed -n '1,12p'
echo "  ..."
unzip -l "$OUT" | tail -3
