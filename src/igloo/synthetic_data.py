"""
Synthetic fitness dataset generator for IGLOO validation.

Three dataset types per protein:
  1. Additive   — BLOSUM-weighted penalty at annotated sites, summed across mutations
  2. Epistatic  — Additive + pairwise interaction terms for co-annotated position pairs
                  (contact weight derived from sequence-distance / shared annotation region)
  3. NonLinear  — Stability-cliff model: fitness collapses when cumulative damage at
                  critical sites exceeds a threshold

Supported proteins (from the table in the design doc):
  - T4Lysozyme  (UniProt P00720, 164 aa)
  - p53DBD      (UniProt P04637, 191-aa DNA-binding domain)
  - GB1         (UniProt P19909, 56-aa B1 domain)

Usage
-----
from src.igloo.synthetic_data import SyntheticDatasetGenerator

gen = SyntheticDatasetGenerator("T4Lysozyme")
df_add  = gen.generate(model="additive",  n_variants=2000, noise=0.1, seed=42)
df_epi  = gen.generate(model="epistatic", n_variants=2000, noise=0.1, seed=42)
df_nl   = gen.generate(model="nonlinear", n_variants=2000, noise=0.1, seed=42)
gen.save_all("data/synthetic")
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

# ---------------------------------------------------------------------------
# BLOSUM62 helper
# ---------------------------------------------------------------------------

_BLOSUM62 = substitution_matrices.load("BLOSUM62")
_BLOSUM62_MIN = min(_BLOSUM62[a][b] for a in _BLOSUM62.alphabet for b in _BLOSUM62.alphabet)
_BLOSUM62_MAX = max(_BLOSUM62[a][b] for a in _BLOSUM62.alphabet for b in _BLOSUM62.alphabet)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def blosum_penalty(wt_aa: str, mut_aa: str) -> float:
    """
    Return a *penalty* (positive = bad) for mutating wt_aa → mut_aa.

    Scaled so that the worst substitution gives 1.0 and a synonymous
    substitution gives 0.0.  Off-diagonal BLOSUM scores are negative
    (unfavourable) or mildly positive (conservative); the diagonal
    (identity) is always the maximum score for that residue.
    """
    if wt_aa == mut_aa:
        return 0.0
    try:
        score = _BLOSUM62[wt_aa][mut_aa]
        diag  = _BLOSUM62[wt_aa][wt_aa]
        # Normalise: penalty = (diag - score) / (diag - global_min)
        penalty = (diag - score) / max(diag - _BLOSUM62_MIN, 1e-9)
        return float(np.clip(penalty, 0.0, 1.0))
    except KeyError:
        return 0.5  # fallback for non-standard amino acids


# ---------------------------------------------------------------------------
# Protein definitions
# ---------------------------------------------------------------------------

@dataclass
class ProteinDefinition:
    """Full specification of one reference protein."""
    name: str
    uniprot_id: str
    sequence: str                       # canonical WT sequence
    # position → annotation category (0-indexed)
    annotations: Dict[int, str] = field(default_factory=dict)
    # category → importance weight
    category_weights: Dict[str, float] = field(default_factory=dict)
    # secondary-structure regions: list of (start, end) ranges, 0-indexed inclusive
    ss_regions: List[Tuple[int, int]] = field(default_factory=list)


# Default category importance weights
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "active_site":    2.0,
    "binding_site":   1.5,
    "metal_binding":  1.2,
    "structural":     0.5,
    "surface":        0.1,
}

# ---------------------------------------------------------------------------
# T4 Lysozyme  (P00720, 164 aa)
# Key sites: Glu11 (proton donor), Asp20 (nucleophile), Trp138 (substrate binding)
# Additional annotated residues from UniProt
# ---------------------------------------------------------------------------
_T4L_SEQ = (
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVD"
    "AALKALTSHHANLKLTTEGETLIQNLKTKGAKDIVDGYGLDNLNALRDYYGCRESMLFLNDRNTLAHRARQL"
    "AINDYIKKMSE"
)  # 164 aa

_T4L_ANNOTATIONS: Dict[int, str] = {
    # Active site
    10: "active_site",   # Glu11 (0-indexed = 10)
    19: "active_site",   # Asp20
    137: "active_site",  # Trp138
    # Binding site residues
    11: "binding_site",
    15: "binding_site",
    17: "binding_site",
    70: "binding_site",
    # Structural / hydrophobic core
    3:  "structural",
    7:  "structural",
    40: "structural",
    99: "structural",
    108: "structural",
    # Surface residues
    0: "surface", 1: "surface", 2: "surface",
    160: "surface", 161: "surface", 162: "surface", 163: "surface",
}

_T4L_SS_REGIONS = [          # approximate helix/sheet regions
    (0, 10),   # N-terminal region
    (11, 35),  # helix A
    (39, 50),  # helix B
    (60, 80),  # helix C
    (81, 100), # helix D
    (101, 130),# helix E
    (131, 155),# helix F
    (156, 163),# C-terminal
]

T4_LYSOZYME = ProteinDefinition(
    name="T4Lysozyme",
    uniprot_id="P00720",
    sequence=_T4L_SEQ,
    annotations={k: v for k, v in _T4L_ANNOTATIONS.items() if 0 <= k < len(_T4L_SEQ)},
    category_weights=_DEFAULT_WEIGHTS.copy(),
    ss_regions=_T4L_SS_REGIONS,
)

# ---------------------------------------------------------------------------
# p53 DNA-Binding Domain  (P04637, residues 94–292 → 191 aa used here)
# Key sites: R248, R273 (DNA contact), R175 (structural/hotspot), Zn-site
# ---------------------------------------------------------------------------
_P53_SEQ = (
    "SVVVPYEPPEVGSDCTTIHYNYMCNSSCMGQMNRRPILTIITLEDSSGKLLGRNSFEVRVCACPGRDRRTEE"
    "ENLRKKGEPVHGQWLDSPRTFMYVPQQHSSEGRAVAIYKQSQHMTEVVRRCPHHERCSEGALPKKMEEHGSE"
    "LEDKGELGWGFHSVVVP"
)  # 191 aa (representative DBD)

# Remap published residue numbers to 0-indexed positions in this fragment
# (published numbers are from full-length p53; offset = 93 for DBD starting at residue 94)
def _p53_pos(full_res: int) -> int:
    return full_res - 94

_P53_ANNOTATIONS: Dict[int, str] = {
    # DNA-contact residues
    _p53_pos(248): "active_site",   # R248 — most mutated hotspot
    _p53_pos(273): "active_site",   # R273
    # Structural hotspots
    _p53_pos(175): "structural",    # R175H common oncogenic mutation
    _p53_pos(249): "structural",    # G249 — local structure
    _p53_pos(245): "structural",    # G245
    # Zinc-binding
    _p53_pos(176): "metal_binding", # Cys176
    _p53_pos(179): "metal_binding", # His179
    _p53_pos(238): "metal_binding", # Cys238
    _p53_pos(242): "metal_binding", # Cys242
    # L3 loop — binding interface
    _p53_pos(237): "binding_site",
    _p53_pos(239): "binding_site",
    _p53_pos(240): "binding_site",
    # Surface
    _p53_pos(94):  "surface",
    _p53_pos(95):  "surface",
    _p53_pos(290): "surface",
    _p53_pos(291): "surface",
    _p53_pos(292): "surface",
}
# Filter out any positions that are out of range for this sequence
_P53_ANNOTATIONS = {k: v for k, v in _P53_ANNOTATIONS.items()
                    if 0 <= k < len(_P53_SEQ)}

_P53_SS_REGIONS = [
    (0, 20),    # beta-sheet S1
    (21, 40),   # loop L1
    (41, 60),   # beta-sheet S2
    (61, 90),   # loop L2 / L3 (contains Zn site)
    (91, 120),  # helix H1
    (121, 150), # beta-sheet S7-S10
    (151, 191), # C-terminal helix H2
]

P53_DBD = ProteinDefinition(
    name="p53DBD",
    uniprot_id="P04637",
    sequence=_P53_SEQ,
    annotations=_P53_ANNOTATIONS,
    category_weights=_DEFAULT_WEIGHTS.copy(),
    ss_regions=_P53_SS_REGIONS,
)

# ---------------------------------------------------------------------------
# GB1 domain  (P19909, 56 aa)
# Key sites: Trp43 (core packing), Asp22, Asn35, Phe30, Tyr33 (IgG interface)
# ---------------------------------------------------------------------------
_GB1_SEQ = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # 56 aa

_GB1_ANNOTATIONS: Dict[int, str] = {
    # IgG-binding interface
    21: "binding_site",  # Asp22
    29: "binding_site",  # Phe30
    32: "binding_site",  # Tyr33
    34: "binding_site",  # Asn35
    42: "active_site",   # Trp43 — hydrophobic core / dual phenotype
    # Structural / beta-sheet core
    3:  "structural",
    5:  "structural",
    20: "structural",
    44: "structural",
    # Surface
    0: "surface", 1: "surface",
    54: "surface", 55: "surface",
}

_GB1_SS_REGIONS = [
    (0, 7),    # beta-strand 1
    (8, 18),   # loop
    (19, 28),  # beta-strand 2
    (29, 39),  # beta-strand 3 / turn
    (40, 46),  # alpha-helix
    (47, 55),  # beta-strand 4
]

GB1 = ProteinDefinition(
    name="GB1",
    uniprot_id="P19909",
    sequence=_GB1_SEQ,
    annotations={k: v for k, v in _GB1_ANNOTATIONS.items() if 0 <= k < len(_GB1_SEQ)},
    category_weights=_DEFAULT_WEIGHTS.copy(),
    ss_regions=_GB1_SS_REGIONS,
)

PROTEINS: Dict[str, ProteinDefinition] = {
    "T4Lysozyme": T4_LYSOZYME,
    "p53DBD":     P53_DBD,
    "GB1":        GB1,
}


# ---------------------------------------------------------------------------
# Variant generator
# ---------------------------------------------------------------------------

def _mutate_sequence(
    sequence: str,
    n_mutations: int,
    rng: np.random.Generator,
) -> Tuple[str, List[Tuple[int, str, str]]]:
    """
    Introduce exactly n_mutations point mutations at distinct positions.

    Returns (mutated_sequence, list_of_(pos, wt_aa, mut_aa)).
    """
    seq = list(sequence)
    positions = rng.choice(len(sequence), size=n_mutations, replace=False)
    mutations = []
    for pos in positions:
        wt_aa = seq[pos]
        choices = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        mut_aa = rng.choice(choices)
        seq[pos] = mut_aa
        mutations.append((int(pos), wt_aa, mut_aa))
    return "".join(seq), mutations


def _sample_n_mutations(rng: np.random.Generator, n_variants: int) -> List[int]:
    """
    Sample mutation counts: 70% single, 20% double, 10% triple.
    """
    counts = rng.choice([1, 2, 3], size=n_variants, p=[0.70, 0.20, 0.10])
    return counts.tolist()


# ---------------------------------------------------------------------------
# Fitness models
# ---------------------------------------------------------------------------

def _importance_weight(protein: ProteinDefinition, pos: int) -> float:
    """Return the ground-truth importance for a position (0 if unannotated)."""
    category = protein.annotations.get(pos)
    if category is None:
        return 0.05  # small background for unannotated positions
    return protein.category_weights.get(category, 0.1)


def _ss_region_of(protein: ProteinDefinition, pos: int) -> Optional[int]:
    """Return the index of the secondary-structure region containing pos, or None."""
    for i, (start, end) in enumerate(protein.ss_regions):
        if start <= pos <= end:
            return i
    return None


def _contact_weight(protein: ProteinDefinition, pos1: int, pos2: int) -> float:
    """
    Sequence-based proxy for structural contact weight.

    Rules (no PDB required):
      - Same SS region → weight 1.0
      - Adjacent SS regions → weight 0.5
      - |pos1 - pos2| ≤ 10 (local neighbourhood) → weight 0.3
      - Otherwise → weight 0.0
    """
    r1 = _ss_region_of(protein, pos1)
    r2 = _ss_region_of(protein, pos2)
    if r1 is not None and r2 is not None:
        if r1 == r2:
            return 1.0
        if abs(r1 - r2) == 1:
            return 0.5
    if abs(pos1 - pos2) <= 10:
        return 0.3
    return 0.0


# -- Model 1: Additive -------------------------------------------------------

def fitness_additive(
    sequence: str,
    protein: ProteinDefinition,
    mutations: List[Tuple[int, str, str]],
) -> float:
    """
    Fitness = 1.0 - Σ importance(pos) * blosum_penalty(wt, mut)
    """
    fitness = 1.0
    for pos, wt_aa, mut_aa in mutations:
        importance = _importance_weight(protein, pos)
        penalty    = importance * blosum_penalty(wt_aa, mut_aa)
        fitness   -= penalty
    return float(fitness)


# -- Model 2: Epistatic -------------------------------------------------------

def fitness_epistatic(
    sequence: str,
    protein: ProteinDefinition,
    mutations: List[Tuple[int, str, str]],
    epistasis_scale: float = 0.1,
) -> float:
    """
    Fitness = additive_fitness
              - epistasis_scale * Σ_{i<j} importance(i)*importance(j)*contact(i,j)
                                          * blosum_penalty(i) * blosum_penalty(j)
    """
    fitness = fitness_additive(sequence, protein, mutations)

    # Pairwise interaction terms
    for k in range(len(mutations)):
        for l in range(k + 1, len(mutations)):
            pos1, wt1, mut1 = mutations[k]
            pos2, wt2, mut2 = mutations[l]
            w1  = _importance_weight(protein, pos1)
            w2  = _importance_weight(protein, pos2)
            cw  = _contact_weight(protein, pos1, pos2)
            if cw == 0.0:
                continue
            p1  = blosum_penalty(wt1, mut1)
            p2  = blosum_penalty(wt2, mut2)
            fitness -= epistasis_scale * w1 * w2 * cw * p1 * p2

    return float(fitness)


# -- Model 3: Non-linear / stability cliff ------------------------------------

def fitness_nonlinear(
    sequence: str,
    protein: ProteinDefinition,
    mutations: List[Tuple[int, str, str]],
    cliff_threshold: float = 1.5,
    cliff_drop: float = 3.0,
) -> float:
    """
    Cumulative damage at *critical* (active_site / metal_binding) positions:
      - Below cliff_threshold → normal additive fitness
      - Above cliff_threshold → sudden stability cliff (fitness collapses)

    Non-critical positions still contribute additively.
    """
    critical_categories = {"active_site", "metal_binding"}

    critical_damage  = 0.0
    noncritical_loss = 0.0

    for pos, wt_aa, mut_aa in mutations:
        importance = _importance_weight(protein, pos)
        penalty    = importance * blosum_penalty(wt_aa, mut_aa)
        category   = protein.annotations.get(pos, "")
        if category in critical_categories:
            critical_damage  += penalty
        else:
            noncritical_loss += penalty

    if critical_damage >= cliff_threshold:
        # Cliff: additive baseline drops sharply
        fitness = 1.0 - noncritical_loss - cliff_drop * critical_damage
    else:
        fitness = 1.0 - noncritical_loss - critical_damage

    return float(fitness)


# ---------------------------------------------------------------------------
# Dataset generator class
# ---------------------------------------------------------------------------

_FITNESS_MODELS = {
    "additive":  fitness_additive,
    "epistatic": fitness_epistatic,
    "nonlinear": fitness_nonlinear,
}

_NOISE_LEVELS = [0.0, 0.1, 0.2, 0.5]


class SyntheticDatasetGenerator:
    """
    Generates synthetic protein fitness datasets for IGLOO validation.

    Parameters
    ----------
    protein_name : str
        One of 'T4Lysozyme', 'p53DBD', 'GB1'.

    Examples
    --------
    >>> gen = SyntheticDatasetGenerator("GB1")
    >>> df  = gen.generate(model="additive", n_variants=500, noise=0.1, seed=0)
    >>> gen.save_all("data/synthetic")
    """

    def __init__(self, protein_name: str) -> None:
        if protein_name not in PROTEINS:
            raise ValueError(
                f"Unknown protein '{protein_name}'. "
                f"Choose from: {list(PROTEINS.keys())}"
            )
        self.protein = PROTEINS[protein_name]
        self._generated: Dict[str, Dict[float, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str = "additive",
        n_variants: int = 2000,
        noise: float = 0.1,
        noise_levels: Optional[List[float]] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset.

        Parameters
        ----------
        model        : 'additive' | 'epistatic' | 'nonlinear'
        n_variants   : number of sequences to generate (WT always included)
        noise        : Gaussian noise std added to fitness (single value)
        noise_levels : if provided, generate one dataset per noise level and
                       return the one matching `noise` (or first if not found)
        seed         : random seed for reproducibility

        Returns
        -------
        pd.DataFrame with columns:
            SEQ, fitness, n_mutations, mutation_string, noise_level,
            model, protein, variant_type
        """
        if model not in _FITNESS_MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(_FITNESS_MODELS.keys())}")

        levels = noise_levels if noise_levels is not None else [noise]
        frames = []
        for lvl in levels:
            df = self._generate_one(model=model, n_variants=n_variants, noise=lvl, seed=seed)
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        # Cache
        if model not in self._generated:
            self._generated[model] = {}
        for lvl in levels:
            self._generated[model][lvl] = combined[combined["noise_level"] == lvl].copy()

        # Return requested noise level
        return combined[combined["noise_level"] == noise].copy().reset_index(drop=True)

    def generate_all(
        self,
        n_variants: int = 2000,
        noise_levels: Optional[List[float]] = None,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all three dataset types at all noise levels.

        Returns dict: model_name → DataFrame (all noise levels concatenated).
        """
        levels = noise_levels if noise_levels is not None else _NOISE_LEVELS
        results = {}
        for model in _FITNESS_MODELS:
            frames = []
            for lvl in levels:
                df = self._generate_one(model=model, n_variants=n_variants, noise=lvl, seed=seed)
                frames.append(df)
            results[model] = pd.concat(frames, ignore_index=True)
        return results

    def save_all(
        self,
        output_dir: str = "data/synthetic",
        n_variants: int = 2000,
        noise_levels: Optional[List[float]] = None,
        seed: int = 42,
    ) -> List[str]:
        """
        Generate all datasets and save as CSVs + ground-truth JSON.

        File naming convention:
            {protein}_{model}_noise{noise}.csv
            {protein}_ground_truth_weights.json

        Returns list of saved file paths.
        """
        levels  = noise_levels if noise_levels is not None else _NOISE_LEVELS
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved   = []

        all_data = self.generate_all(n_variants=n_variants, noise_levels=levels, seed=seed)

        for model, df in all_data.items():
            for lvl in levels:
                subset = df[df["noise_level"] == lvl].copy().reset_index(drop=True)
                fname  = out_dir / f"{self.protein.name}_{model}_noise{lvl:.1f}.csv"
                subset.to_csv(fname, index=False)
                print(f"  Saved {fname}  ({len(subset)} rows)")
                saved.append(str(fname))

        # Ground-truth importance vector
        gt_path = out_dir / f"{self.protein.name}_ground_truth_weights.json"
        self._save_ground_truth(gt_path)
        print(f"  Saved {gt_path}")
        saved.append(str(gt_path))

        return saved

    def ground_truth_importance(self) -> np.ndarray:
        """
        Return a 1-D importance vector of length len(sequence).
        Position i = weight assigned to that site (0.05 if unannotated).
        """
        n = len(self.protein.sequence)
        weights = np.full(n, 0.05)
        for pos, category in self.protein.annotations.items():
            if 0 <= pos < n:
                weights[pos] = self.protein.category_weights.get(category, 0.05)
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        model: str,
        n_variants: int,
        noise: float,
        seed: int,
    ) -> pd.DataFrame:
        rng       = np.random.default_rng(seed + int(noise * 1000))
        wt_seq    = self.protein.sequence
        fit_fn    = _FITNESS_MODELS[model]
        mut_counts = _sample_n_mutations(rng, n_variants)

        rows = []

        # Always include wild-type
        rows.append(self._wt_row(wt_seq, model, noise))

        seen_seqs = {wt_seq}

        for n_mut in mut_counts:
            # Retry up to 10 times to get a unique sequence
            for _ in range(10):
                seq, mutations = _mutate_sequence(wt_seq, n_mut, rng)
                if seq not in seen_seqs:
                    break
            seen_seqs.add(seq)

            raw_fitness = fit_fn(seq, self.protein, mutations)
            noisy_fitness = raw_fitness + (rng.normal(0, noise) if noise > 0 else 0.0)

            mut_str = ";".join(f"{wt}{pos+1}{mut}" for pos, wt, mut in mutations)
            variant_type = {1: "single", 2: "double", 3: "triple"}.get(n_mut, "multi")

            rows.append({
                "SEQ":            seq,
                "fitness":        round(noisy_fitness, 6),
                "fitness_noiseless": round(raw_fitness, 6),
                "n_mutations":    n_mut,
                "mutation_string": mut_str,
                "noise_level":    noise,
                "model":          model,
                "protein":        self.protein.name,
                "variant_type":   variant_type,
                "is_wildtype":    False,
            })

        return pd.DataFrame(rows)

    def _wt_row(self, wt_seq: str, model: str, noise: float) -> dict:
        return {
            "SEQ":               wt_seq,
            "fitness":           1.0,
            "fitness_noiseless": 1.0,
            "n_mutations":       0,
            "mutation_string":   "WT",
            "noise_level":       noise,
            "model":             model,
            "protein":           self.protein.name,
            "variant_type":      "wildtype",
            "is_wildtype":       True,
        }

    def _save_ground_truth(self, path: Path) -> None:
        weights = self.ground_truth_importance()
        data = {
            "protein":          self.protein.name,
            "uniprot_id":       self.protein.uniprot_id,
            "sequence_length":  len(self.protein.sequence),
            "category_weights": self.protein.category_weights,
            "annotations": {
                str(pos): cat for pos, cat in self.protein.annotations.items()
            },
            "importance_vector": weights.tolist(),
            "annotated_positions": {
                cat: sorted([p for p, c in self.protein.annotations.items() if c == cat])
                for cat in set(self.protein.annotations.values())
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def generate_all_proteins(
    output_dir: str = "data/synthetic",
    n_variants: int = 2000,
    noise_levels: Optional[List[float]] = None,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Generate all three dataset types for all three proteins and save to disk.

    Returns dict: protein_name → list of saved file paths.
    """
    levels  = noise_levels if noise_levels is not None else _NOISE_LEVELS
    results = {}
    for name in PROTEINS:
        print(f"\n{'='*60}")
        print(f"Generating synthetic datasets for {name}")
        print(f"{'='*60}")
        gen   = SyntheticDatasetGenerator(name)
        saved = gen.save_all(output_dir=output_dir, n_variants=n_variants,
                             noise_levels=levels, seed=seed)
        results[name] = saved
    return results
