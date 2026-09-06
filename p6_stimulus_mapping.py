"""P6 signed keyword routing (not a validated natural-language understanding model)."""
import re
import unicodedata
from dataclasses import asdict

import torch
from engine_v2_corrected import TensionEngine, EngineConfig, DEVICE
from report_utils import export_run

POLARITY_POLES = (
    ("Poder", "Vulnerabilidad"), ("Placer", "Dolor"), ("Integración", "Fragmentación"),
    ("Control", "Rendición"), ("Deseo", "Límite"), ("Libertad", "Orden"),
    ("Preservación", "Transformación"), ("Reconocimiento", "Autenticidad"),
)
NODES = [" vs ".join(pair) for pair in POLARITY_POLES]

def _normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold())
                   if not unicodedata.combining(c))

def polarity_evidence(text):
    """Separate observed positive/negative keyword evidence for all eight types."""
    normalized = _normalize(text)
    evidence = []
    for positive, negative in POLARITY_POLES:
        evidence.append(tuple(bool(re.search(r"\b" + re.escape(_normalize(pole)) + r"\b", normalized))
                              for pole in (positive, negative)))
    if not any(any(pair) for pair in evidence):
        raise ValueError("No recognized polarity pole; provide an explicit signed stimulus")
    return evidence

def stim_from_text(text, N, scale=0.5):
    if not isinstance(N, int) or N < 1 or not 0 < scale <= 1:
        raise ValueError("N must be positive and 0 < scale <= 1")
    evidence = polarity_evidence(text)
    stimulus = torch.zeros(N, device=DEVICE)
    for i, (positive, negative) in enumerate(evidence):
        # Two observed poles have zero net signed input, not evidence of inactivity.
        stimulus[i::len(POLARITY_POLES)] = scale * (int(positive) - int(negative))
    return stimulus

def main(prompt="Deseo", out_dir="results_corrected/p6_stimulus",
         steps=20, N=160, K=10, seed=2025, scale=0.5, stim_step=8, plots=True):
    if not 0 <= stim_step < steps:
        raise ValueError("stim_step must address an executed zero-indexed step")
    cfg = EngineConfig(N=N, K=K, steps=steps, seed=seed, stim_step=stim_step,
                       stim_scale=scale, learn_M=False)
    stimulus = stim_from_text(prompt, N, scale)
    eng = TensionEngine(cfg, nodes=[NODES[i % len(NODES)] for i in range(N)])
    for t in range(steps):
        eng.step(stimulus=stimulus if t == stim_step else None)
    control = TensionEngine(cfg)
    control.run(print_console=False)
    metadata = {"experiment": "P6", "config": asdict(cfg), "seed": seed, "prompt": prompt,
                "mapping": "explicit_signed_keyword_v1", "pole_evidence": polarity_evidence(prompt),
                "step_indexing": "zero_based", "stim_step": stim_step, "human_step": stim_step + 1,
                "polarity_types": len(POLARITY_POLES), "state_units": N,
                "ambiguity": "Both pole keywords are recorded independently but cancel in legacy signed projection."}
    return export_run(out_dir, "p6_stimulus", eng.ts, f"P6 signed input: {prompt}",
                      metadata=metadata, control=control.ts, plots=plots)

if __name__ == "__main__":
    main()
