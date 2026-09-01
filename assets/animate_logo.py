#!/usr/bin/env python3
"""Render light and dark palindromic fitting/unfitting logo GIFs."""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SVG_NS = "http://www.w3.org/2000/svg"
NS = {"svg": SVG_NS}
ET.register_namespace("", SVG_NS)


@dataclass(frozen=True)
class Theme:
    """Colours that must change with the GIF canvas."""

    background: str
    foreground: str
    axes_opacity: str


THEMES = {
    "light": Theme("#FFFFFF", "#2D2B3A", "0.38"),
    "dark": Theme("#0D1117", "#F0F6FC", "0.48"),
}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def smoothstep(value: float) -> float:
    value = clamp(value)
    return value * value * (3.0 - 2.0 * value)


def phase(time: float, start: float, end: float) -> float:
    return smoothstep((time - start) / (end - start))


def group(root: ET.Element, element_id: str) -> ET.Element:
    element = root.find(f".//svg:g[@id='{element_id}']", NS)
    if element is None:
        raise RuntimeError(f"Missing SVG group #{element_id}")
    return element


def add_reveal_clip(
    root: ET.Element, clip_id: str, progress: float
) -> None:
    defs = root.find("svg:defs", NS)
    if defs is None:
        defs = ET.Element(f"{{{SVG_NS}}}defs")
        root.insert(2, defs)

    clip = ET.SubElement(defs, f"{{{SVG_NS}}}clipPath", {"id": clip_id})
    ET.SubElement(
        clip,
        f"{{{SVG_NS}}}rect",
        {
            "x": "90",
            "y": "140",
            "width": f"{348.0 * progress:.3f}",
            "height": "310",
        },
    )


def build_frame(
    template: ET.Element, progress: float, theme: Theme
) -> ET.Element:
    root = copy.deepcopy(template)

    # GIF transparency produces jagged antialiasing. Render each theme against the
    # same opaque canvas that will surround it on GitHub instead.
    background = ET.Element(
        f"{{{SVG_NS}}}rect",
        {
            "x": "60",
            "y": "60",
            "width": "410",
            "height": "410",
            "fill": theme.background,
        },
    )
    root.insert(2, background)

    axes = group(root, "axes")
    envelopes = group(root, "envelopes")
    fit = group(root, "fit")
    samples = group(root, "samples")

    # The empty coordinate system is the permanent first and last state.
    axes.set("fill", theme.foreground)
    axes.set("opacity", theme.axes_opacity)

    # The first fit polygon is a narrow knockout around the purple curve. The
    # two sample layers similarly form foreground rings with canvas-coloured
    # centres. Matching these to the theme keeps their contrast consistent.
    fit_layers = fit.findall("svg:polygon", NS)
    if len(fit_layers) < 2:
        raise RuntimeError("Missing SVG fit polygons")
    fit_layers[0].set("fill", theme.background)

    sample_layers = samples.findall("svg:g", NS)
    if len(sample_layers) < 2:
        raise RuntimeError("Missing SVG sample layers")
    sample_layers[0].set("fill", theme.foreground)
    sample_layers[1].set("fill", theme.background)

    envelope_progress = phase(progress, 0.00, 0.48)
    fit_progress = phase(progress, 0.15, 0.72)
    add_reveal_clip(root, "envelope-reveal", envelope_progress)
    add_reveal_clip(root, "fit-reveal", fit_progress)
    envelopes.set("clip-path", "url(#envelope-reveal)")
    fit.set("clip-path", "url(#fit-reveal)")

    for layer in sample_layers:
        for index, circle in enumerate(layer.findall("svg:circle", NS)):
            start = 0.54 + 0.026 * index
            circle.set(
                "opacity", f"{phase(progress, start, start + 0.060):.4f}"
            )

    return root


def palindromic_progress(frame_count: int) -> list[float]:
    """Return 0 -> 1 -> 0 progress, including both empty-axis endpoints."""
    if frame_count < 5 or frame_count % 2 == 0:
        raise ValueError("--frames must be an odd integer of at least 5")

    half = frame_count // 2
    forward = [index / half for index in range(half + 1)]
    reverse = [index / half for index in range(half - 1, -1, -1)]

    # Small endpoint plateaus make both the empty axes and completed fit readable.
    return [phase(value, 0.05, 0.92) for value in forward + reverse]


def render(
    source: Path,
    output: Path,
    size: int,
    frames: int,
    delay: int,
    theme: Theme,
) -> None:
    magick = shutil.which("magick")
    if magick is None:
        raise RuntimeError("ImageMagick's `magick` executable is required")

    template = ET.parse(source).getroot()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="memory-kernel-logo-") as directory:
        frame_dir = Path(directory)
        png_frames: list[Path] = []

        timeline = palindromic_progress(frames)
        for index, progress in enumerate(timeline):
            svg_path = frame_dir / f"frame-{index:03d}.svg"
            png_path = frame_dir / f"frame-{index:03d}.png"
            ET.ElementTree(build_frame(template, progress, theme)).write(
                svg_path, encoding="utf-8", xml_declaration=True
            )
            subprocess.run(
                [
                    magick,
                    "-background",
                    theme.background,
                    str(svg_path),
                    "-alpha",
                    "remove",
                    "-alpha",
                    "off",
                    "-resize",
                    f"{size}x{size}",
                    str(png_path),
                ],
                check=True,
            )
            png_frames.append(png_path)

        subprocess.run(
            [
                magick,
                "-delay",
                str(delay),
                *map(str, png_frames),
                "-loop",
                "0",
                "-dither",
                "None",
                "-colors",
                "96",
                str(output),
            ],
            check=True,
        )


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=directory / "logo-frameless.svg"
    )
    parser.add_argument(
        "--output", type=Path, default=directory / "logo-frameless.gif"
    )
    parser.add_argument(
        "--theme",
        choices=("both", *THEMES),
        default="both",
        help=(
            "Theme to render. 'both' writes --output for light mode and a "
            "-dark sibling for dark mode"
        ),
    )
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument(
        "--frames",
        type=int,
        default=73,
        help="Odd total frame count for the forward-and-reverse loop",
    )
    parser.add_argument(
        "--delay", type=int, default=6, help="Frame delay in centiseconds"
    )
    arguments = parser.parse_args()
    selected_themes = THEMES if arguments.theme == "both" else {
        arguments.theme: THEMES[arguments.theme]
    }
    for name, theme in selected_themes.items():
        output = arguments.output
        if arguments.theme == "both" and name == "dark":
            output = output.with_name(f"{output.stem}-dark{output.suffix}")
        render(
            arguments.source,
            output,
            arguments.size,
            arguments.frames,
            arguments.delay,
            theme,
        )


if __name__ == "__main__":
    main()
