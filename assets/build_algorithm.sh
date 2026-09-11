#!/usr/bin/env bash
# Build assets/gle-baoab{,-dark}.svg from gle_baoab_algorithm.tex.
#
#   ./assets/build_algorithm.sh
#
# Needs a TeX installation with algorithm2e, newtx and inconsolata, plus dvisvgm.
#
# The DVI route is deliberate: dvisvgm's --pdf mode needs mutool, which is a separate
# install, while DVI needs nothing extra. \pagecolor arrives as a full-size background
# rect, which is what keeps the figure legible on GitHub's dark ground.
#
# --no-fonts traces the glyphs into paths. GitHub serves the SVG through <img>, where
# font loading is blocked, so an SVG that references fonts would render in a fallback
# face with the wrong metrics.
set -euo pipefail

cd "$(dirname "$0")"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

build() {                       # build <output-stem> [tex-preamble]
    local stem="$1" pre="${2:-}"
    local input="gle_baoab_algorithm.tex"
    [ -n "$pre" ] && input="${pre}\\input{gle_baoab_algorithm}"
    latex -interaction=nonstopmode -halt-on-error \
          -output-directory="$work" -jobname="$stem" "$input" > "$work/$stem.log"
    dvisvgm --no-fonts --exact --bbox=preview \
            --output="$work/$stem.svg" "$work/$stem.dvi" > /dev/null 2>&1
    python3 - "$work/$stem.svg" "$stem.svg" <<'PY'
import re, sys
src, dst = sys.argv[1], sys.argv[2]
svg = open(src, encoding="utf-8").read()
title = "gle-BAOAB, one step"
desc = ("One step of the quasi-Markovian generalised Langevin equation: a force half-step, "
        "a drift half-step, an exact Ornstein-Uhlenbeck step for the auxiliary variables at "
        "frozen configuration, then the drift and force half-steps again. Scheme (B.1) of "
        "Sachs' thesis, with the mass matrix restored.")
svg = re.sub(r"<svg ([^>]*)>",
             lambda m: f'<svg {m.group(1)} role="img" aria-labelledby="alg-title alg-desc">'
                       f'\n<title id="alg-title">{title}</title>'
                       f'\n<desc id="alg-desc">{desc}</desc>',
             svg, count=1)
open(dst, "w", encoding="utf-8").write(svg)
print("wrote", dst)
PY
}

build gle-baoab
build gle-baoab-dark '\def\DARKTHEME{}'
