#!/usr/bin/env python3
"""Generate the module map in assets/module-map{,-dark}.svg.

The map shows how a fitted Prony series reaches NQCDynamics' GLE integrator: the
offline fitting chain, the adapter in scripts/gle/prony_memory.jl, and the
NQCDynamics modules it plugs into. Two files are emitted because GitHub picks
between them with <picture media="(prefers-color-scheme: dark)">, the same way
the logo does.

    python3 assets/module_map.py
"""

from pathlib import Path

LIGHT = dict(bg="#FFFFFF", edge_bg="#D9DEE6", surface="#FFFFFF", ink="#1B2130",
             muted="#5F6A7D", fit="#95570F", bridge="#7446A8", dyn="#1D6A87")
DARK = dict(bg="#0D1117", edge_bg="#2C323D", surface="#161B22", ink="#E6E9EF",
            muted="#97A0AF", fit="#E2A65A", bridge="#B996EC", dyn="#63B6D3")

# (x, y, w, h, stroke-class)
BOXES = [
    (150, 40, 184, 100, "s-fit"), (358, 40, 184, 100, "s-fit"),
    (566, 40, 184, 100, "s-fit"), (774, 40, 184, 100, "s-fit"),
    (982, 40, 184, 100, "s-data"),
    (150, 250, 184, 100, "s-bridge"), (358, 250, 184, 100, "s-bridge"),
    (566, 250, 184, 100, "s-bridge"), (774, 250, 184, 100, "s-bridge"),
    (982, 250, 184, 100, "s-bridge"),
    (150, 460, 184, 100, "s-dyn"), (358, 460, 184, 100, "s-dyn"),
    (566, 460, 392, 100, "s-dyn"), (982, 460, 184, 100, "s-dyn"),
]

# (x, y, class, anchor, text)
TEXTS = [
    # lane labels
    (12, 76, "lane fit", "start", "MemoryKernel"),
    (12, 92, "lane fit", "start", "Fitting.jl"),
    (12, 108, "lane-sub", "start", "src/ (package)"),
    (12, 286, "lane bridge", "start", "Bridge"),
    (12, 302, "lane-sub", "start", "scripts/gle/"),
    (12, 317, "lane-sub", "start", "prony_memory.jl"),
    (12, 496, "lane dyn", "start", "NQCDynamics.jl"),
    (12, 512, "lane-sub", "start", "GLE_MDEF branch"),
    (12, 527, "lane-sub", "start", "DynamicsMethods/"),
    # band 1
    (160, 62, "ttl fit", "start", "SpectralData"),
    (160, 82, "ln", "start", "load_lambda_spectrum"),
    (160, 97, "ln", "start", "ErpenbeckThossModel"),
    (160, 112, "ln", "start", "h(x), Δ(x)"),
    (160, 127, "ln", "start", "markovian_friction(x)"),
    (368, 62, "ttl fit", "start", "PronySeriesFriction"),
    (368, 82, "ln", "start", "PoleSet(λ, α, β)"),
    (368, 97, "ln", "start", "ContinuousAmplitudes"),
    (368, 112, "ln", "start", "qgle_blocks → Γ₂₂"),
    (368, 127, "ln", "start", "qgle_coupling! → g, a₀"),
    (576, 62, "ttl fit", "start", "Objective"),
    (576, 82, "ln", "start", "design_matrix"),
    (576, 97, "ln", "start", "residual_scales"),
    (576, 112, "ln", "start", "configuration_weights"),
    (576, 127, "ln", "start", "objective_value → J"),
    (784, 62, "ttl fit", "start", "Fitting"),
    (784, 82, "ln", "start", "fit_amplitudes"),
    (784, 97, "lbl", "start", "  stage 1: NNLS per x"),
    (784, 112, "ln", "start", "fit_basis_report"),
    (784, 127, "lbl", "start", "  stage 2: Chebyshev"),
    (992, 62, "ttl ink", "start", "data/fits/*.h5"),
    (992, 82, "ln", "start", "prony_stage2Cheby_…"),
    (992, 97, "ln", "start", "C_cheb"),
    (992, 112, "ln", "start", "lambda/alpha/beta_Ha"),
    (992, 127, "ln", "start", "K_R, K_C"),
    # band 1 edge labels
    (346, 30, "lbl", "middle", "h, Δ, Λ(x,0)"),
    (554, 30, "lbl", "middle", "Λ_θ(x, ω)"),
    (762, 30, "lbl", "middle", "J(θ)"),
    (970, 30, "lbl", "middle", "writes (scripts/fitting)"),
    # band 1 -> band 2 edge labels
    (250, 200, "lbl", "start", "markovian_friction(x)"),
    (458, 226, "lbl bridge", "start", "qgle_coupling!(x) → g, a₀"),
    (673, 165, "lbl", "middle", "ContinuousAmplitudes, qgle_blocks → Γ₂₂"),
    (980, 189, "lbl", "middle", "h5open, once"),
    # band 2
    (160, 272, "ttl bridge", "start", "PES + MDEF ref."),
    (160, 292, "ln", "start", "GroundAdiabat"),
    (160, 307, "ln", "start", "erpenbeck_thoss_neutral"),
    (160, 322, "ln", "start", "MarkovianFriction"),
    (160, 337, "lbl", "start", "  → Λ(x,0) for MDEF"),
    (368, 272, "ttl bridge", "start", "GLE_friction!"),
    (368, 292, "ln", "start", "x = r · BOHR_IN_ANGSTROM"),
    (368, 307, "ln", "start", "Γ₁₁ = a₀ / m"),
    (368, 322, "ln", "start", "Γ₁₂ = −gᵀ / √m"),
    (368, 337, "ln", "start", "Γ₂₁ =  g  / √m"),
    (576, 272, "ttl bridge", "start", "prony_simulation"),
    (576, 292, "ln", "start", "Simulation{GLE_MDEF}"),
    (576, 307, "ln", "start", "method.Λ .= Γ₂₂"),
    (576, 322, "ln", "start", "Γ[2:end,2:end] .= Γ₂₂"),
    (576, 337, "lbl", "start", "  requires d = 1"),
    (784, 272, "ttl bridge", "start", "PronyMemory"),
    (784, 292, "ln", "start", "<: MemoryModel"),
    (784, 307, "ln", "start", "c   g(x) buffer, m = 43"),
    (784, 322, "ln", "start", "Γ₂₂ fixed (regime A)"),
    (784, 337, "ln", "start", "model::Continuous…"),
    (992, 272, "ttl bridge", "start", "SeededIC"),
    (992, 292, "ln", "start", "seed_auxiliaries!"),
    (992, 307, "ln", "start", "s(0) ~ N(0, kT·I)"),
    (992, 322, "lbl", "start", "  keyed on trajectory i"),
    (992, 337, "ln", "start", "writes method.s"),
    # band 2 -> band 3 edge labels
    (250, 400, "lbl", "start", "ClassicalModel,"),
    (250, 414, "lbl", "start", "FrictionProvider"),
    (458, 400, "lbl bridge", "start", "calls f.g every step"),
    (458, 414, "lbl bridge", "start", "at the midpoint q(k+½)"),
    (666, 400, "lbl", "start", "Simulation{GLE_MDEF}"),
    (666, 414, "lbl", "start", "fills Λ, Γ₂₂ once"),
    (874, 388, "lbl", "start", "<: MemoryModel"),
    (874, 402, "lbl", "start", "(dispatch key)"),
    (1066, 428, "lbl", "end", "sample_distribution"),
    (1066, 442, "lbl", "end", "on each copy, index i"),
    # band 3
    (160, 482, "ttl dyn", "start", "NQCModels"),
    (160, 502, "lbl", "start", "+ NQCCalculators"),
    (160, 517, "ln", "start", "ClassicalModel"),
    (160, 532, "ln", "start", "FrictionProvider"),
    (160, 547, "ln", "start", "cache, update_cache!"),
    (368, 482, "ttl dyn", "start", "IntegrationAlgorithms"),
    (368, 502, "ln", "start", "GLE_BAOAB  B·A·O·A·B"),
    (368, 517, "ln", "start", "GLE_step_O!: e^(−ΔtΓ)"),
    (368, 532, "ln", "start", "S Sᵀ = I − F Fᵀ"),
    (368, 547, "lbl", "start", "exact OU step"),
    (576, 482, "ttl dyn", "start", "ClassicalMethods"),
    (576, 502, "lbl", "start", "gle_mdef.jl"),
    (576, 517, "ln", "start", "GLE_MDEF{m,T,MM}"),
    (576, 532, "ln", "start", "Γ, Λ, C, s live here"),
    (576, 547, "ln", "start", "create_problem"),
    (770, 502, "lbl", "start", "mdef.jl"),
    (770, 517, "ln", "start", "MDEF, friction!"),
    (770, 532, "ln", "start", "ConstantProny"),
    (770, 547, "ln", "start", "ConfigurationDependent…"),
    (992, 482, "ttl dyn", "start", "Ensembles"),
    (992, 502, "ln", "start", "run_dynamics"),
    (992, 517, "ln", "start", "OrderedSelection"),
    (992, 532, "ln", "start", "sample_distribution"),
    (992, 547, "ln", "start", "EnsembleProblem"),
    # band 3 internal
    (554, 453, "lbl", "middle", "reads Γ, writes s"),
    (762, 591, "lbl", "middle",
     "solve(EnsembleProblem, GLE_BAOAB) · sim deep-copied per trajectory"),
]

# (kind, geometry, class, marker-start, marker-end)
EDGES = [
    ("line", (334, 90, 358, 90), "edge", False, True),
    ("line", (542, 90, 566, 90), "edge", False, True),
    ("line", (750, 90, 774, 90), "edge", False, True),
    ("line", (958, 90, 982, 90), "edge", False, True),
    ("line", (242, 140, 242, 250), "edge", False, True),
    ("line", (450, 140, 450, 250), "edge-hot", False, True),
    ("poly", "500,140 500,172 846,172 846,250", "edge", False, True),
    ("poly", "1074,140 1074,196 886,196 886,250", "edge", False, True),
    ("line", (242, 350, 242, 460), "edge", False, True),
    ("line", (450, 460, 450, 350), "edge-hot", False, True),
    ("line", (658, 350, 658, 460), "edge", False, True),
    ("line", (866, 350, 866, 460), "edge", False, True),
    ("line", (1074, 350, 1074, 460), "edge", False, True),
    ("line", (542, 510, 566, 510), "edge", True, True),
    ("poly", "1074,560 1074,598 450,598 450,560", "edge", False, True),
]

TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 612" width="1180" \
role="img" aria-labelledby="mm-title mm-desc">
<title id="mm-title">From a fitted Prony series to GLE dynamics</title>
<desc id="mm-desc">MemoryKernelFitting's modules produce a stage-2 Chebyshev fit file. The adapter \
in scripts/gle/prony_memory.jl reads it into a PronyMemory memory model and builds a \
Simulation of GLE_MDEF. Every integrator step, NQCDynamics calls the adapter's \
GLE_friction! at the half-step position, which rebuilds the coupling vector through \
qgle_coupling! and writes it into the drift matrix Gamma.</desc>
<style>
text {{ font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.ttl {{ font-size: 12.5px; font-weight: 600; }}
.ln {{ font-size: 11px; fill: {ink}; }}
.lbl {{ font-size: 11px; fill: {muted}; }}
.lane {{ font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
        font-size: 12.5px; font-weight: 600; }}
.lane-sub {{ font-size: 11px; fill: {muted}; }}
.fit {{ fill: {fit}; }} .bridge {{ fill: {bridge}; }} .dyn {{ fill: {dyn}; }} .ink {{ fill: {ink}; }}
.box {{ fill: {surface}; stroke-width: 1.25; }}
.s-fit {{ stroke: {fit}; }} .s-bridge {{ stroke: {bridge}; }} .s-dyn {{ stroke: {dyn}; }}
.s-data {{ stroke: {muted}; stroke-dasharray: 5 4; }}
.edge {{ fill: none; stroke: {muted}; stroke-width: 1.25; }}
.edge-hot {{ fill: none; stroke: {bridge}; stroke-width: 2; }}
</style>
<defs>
<marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" \
markerWidth="8" markerHeight="8" orient="auto-start-reverse">\
<path d="M0,0 L10,5 L0,10 z" fill="{muted}"/></marker>
<marker id="ahh" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" \
markerWidth="9" markerHeight="9" orient="auto-start-reverse">\
<path d="M0,0 L10,5 L0,10 z" fill="{bridge}"/></marker>
</defs>
<rect x="1" y="1" width="1178" height="610" rx="8" fill="{bg}" stroke="{edge_bg}"/>
{body}
</svg>
"""


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(palette):
    parts = []
    for x, y, w, h, cls in BOXES:
        parts.append(f'<rect class="box {cls}" x="{x}" y="{y}" width="{w}" '
                     f'height="{h}" rx="4"/>')
    for kind, geo, cls, ms, me in EDGES:
        mk = (' marker-start="url(#ahh)"' if ms else "") if cls == "edge-hot" else \
             (' marker-start="url(#ah)"' if ms else "")
        mk += (' marker-end="url(#ahh)"' if me else "") if cls == "edge-hot" else \
              (' marker-end="url(#ah)"' if me else "")
        if kind == "line":
            x1, y1, x2, y2 = geo
            parts.append(f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" '
                         f'y2="{y2}"{mk}/>')
        else:
            parts.append(f'<polyline class="{cls}" points="{geo}"{mk}/>')
    for x, y, cls, anchor, txt in TEXTS:
        a = "" if anchor == "start" else f' text-anchor="{anchor}"'
        parts.append(f'<text class="{cls}" x="{x}" y="{y}"{a}>{esc(txt)}</text>')
    return TEMPLATE.format(body="\n".join(parts), **palette)


if __name__ == "__main__":
    here = Path(__file__).parent
    (here / "module-map.svg").write_text(render(LIGHT), encoding="utf-8")
    (here / "module-map-dark.svg").write_text(render(DARK), encoding="utf-8")
    print("wrote module-map.svg and module-map-dark.svg")
