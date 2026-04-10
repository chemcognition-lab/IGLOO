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
  - p53         (UniProt P04637, full-length 393 aa)
  - GB1         (UniProt P06654, 56-aa B1 domain)

Usage
-----
from src.igloo.synthetic_data import SyntheticDatasetGenerator

gen = SyntheticDatasetGenerator("T4Lysozyme")
df_add  = gen.generate(model="additive",  n_variants=2000, seed=42)
df_epi  = gen.generate(model="epistatic", n_variants=2000, seed=42)
df_nl   = gen.generate(model="nonlinear", n_variants=2000, seed=42)
gen.save_all("data/synthetic")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
    annotations: dict[int, str] = field(default_factory=dict)
    # category → importance weight
    category_weights: dict[str, float] = field(default_factory=dict)
    # secondary-structure regions: list of (start, end) ranges, 0-indexed inclusive
    ss_regions: list[tuple[int, int]] = field(default_factory=list)
    # recommended number of variants (based on single-mutant space coverage)
    default_n_variants: int = 2000


# Per-protein category importance weights
# Values reflect the biological role of each category within that protein's function.
_T4L_WEIGHTS: dict[str, float] = {
    "active_site": 5, # Glu11/Asp20 — catalytic, complete loss of activity when mutated
    "binding_site": 1.5, # substrate binding residues
    "structural": 0.5, # salt bridges (His31, Asp70) — stability contribution
    }

_P53_WEIGHTS: dict[str, float] = {
    "active_site": 1.5, # direct DNA contacts (R248, R273, K120, C277, R280) — oncogenic hotspots
    "metal_binding": 5, # Zn coordination (C176, H179, C238, C242) — loss unfolds DBD
    "binding_site": 0.5, # MDM2 interface, polymerization, DNA backbone contacts
    "structural": 1.0, # TET hinge/salt bridge — affects tetramerization
    "signalling": 0.5, # NLS residues — affects localization, less directly lethal
    }

_GB1_WEIGHTS: dict[str, float] = {
    "binding_site": 1.5, # Fc/Fab interface — primary function of the domain
    "structural": 1.0, # core packing — critical for stability of this 56-aa domain
    "folding": 0.5, # beta-hairpin residues — important for folding/stability
    }

# ---------------------------------------------------------------------------
# T4 Lysozyme  (P00720, 164 aa)
# Key sites: Glu11 (catalysis), Asp20 (catalysis), His26 (binding/stereochemistry),
#            His31 (structural/salt bridge), Asp70 (structural/salt bridge),
#            Ser117 (substrate binding), Asn132 (substrate binding)
# Secondary structure: Helix A, Beta Strands 1-3, Helices B-J
# ---------------------------------------------------------------------------
_T4L_SEQ = (
    "MNIFEMLRIDEGLRLKIYKDTEGYYHIGIGHLLTKSPSLNAAKSELDKAIGRNCNGVITKDEAEKLFNQDVD"
    "AAVRGILRNAKLKPVYDSLDAVRRCALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTP"
    "NRAKRVITTFRTGTWDAYKNL"
)  # 164 aa

_T4L_ANNOTATIONS: dict[int, str] = {
    # Active site (catalysis)
    10: "active_site",   # Glu11 (0-indexed = 10) — catalysis
    19: "active_site",   # Asp20 — catalysis
    # Binding site residues
    25: "binding_site",  # His26 — H26T changes product stereochemistry
    116: "binding_site", # Ser117 — substrate binding, mutation to V removes activity
    131: "binding_site", # Asn132 — substrate binding
    # Structural (salt bridges)
    30: "structural",   # His31 — salt bridge
    69: "structural",   # Asp70 — salt bridge
}

_T4L_SS_REGIONS = [           # 0-indexed from spreadsheet (1-indexed positions - 1)
    (1, 11),    # Helix A        (positions 2–12)
    (13, 18),   # Beta Strand 1  (positions 14–19)
    (24, 27),   # Beta Strand 2  (positions 25–28)
    (30, 33),   # Beta Strand 3  (positions 31–34)
    (37, 50),   # Helix B        (positions 38–51)
    (58, 79),   # Helix C        (positions 59–80)
    (82, 89),   # Helix D        (positions 83–90)
    (92, 105),  # Helix E        (positions 93–106)
    (107, 112), # Helix F        (positions 108–113)
    (114, 122), # Helix G        (positions 115–123)
    (125, 133), # Helix H        (positions 126–134)
    (136, 140), # Helix I        (positions 137–141)
    (142, 154), # Helix J        (positions 143–155)
]

T4_LYSOZYME = ProteinDefinition(
    name="T4Lysozyme",
    uniprot_id="P00720",
    sequence=_T4L_SEQ,
    annotations={k: v for k, v in _T4L_ANNOTATIONS.items() if 0 <= k < len(_T4L_SEQ)},
    category_weights=_T4L_WEIGHTS.copy(),
    ss_regions=_T4L_SS_REGIONS,
    default_n_variants=4000,  # ~36% single-mutant coverage (space=3,116)
)

# ---------------------------------------------------------------------------
# p53  (P04637, full-length 393 aa)
# Key sites: F19/L22/W23/L26 (MDM2 binding), K120/R248/R273/C277/R280 (DNA),
#            C176/H179/C238/C242 (Zn coordination), G334/R337 (TET structural),
#            NLS1 (K319-K321), NLS2 (K370-K373), NLS3 (K381-K382)
# Domains: TAD1 (1-40), TAD2 (41-61), PRD (62-92), DBD (96-292),
#           IDR (293-323), TET (324-356), CTD (364-393)
# ---------------------------------------------------------------------------
_P53_SEQ = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"
    "DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKS"
    "VTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSD"
    "SDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMN"
    "RRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNN"
    "TSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQ"
    "STSRHKKLMFKTEGPDSD"
)  # 393 aa

_P53_ANNOTATIONS: dict[int, str] = {
    # MDM2 binding interface (TAD1)
    18: "binding_site",   # F19
    21: "binding_site",   # L22
    22: "binding_site",   # W23
    25: "binding_site",   # L26
    # DNA binding (DBD)
    106: "binding_site",  # Y107 — polymerization
    119: "active_site",   # K120 — DNA major groove
    # Zinc coordination (DBD)
    175: "metal_binding", # C176
    177: "binding_site",  # H178 — polymerization
    178: "metal_binding", # H179
    237: "metal_binding", # C238
    240: "binding_site",  # S241 — DNA backbone
    241: "metal_binding", # C242
    # DNA contact residues (DBD)
    242: "active_site",   # M243 — DNA contact
    243: "active_site",   # G244 — DNA contact
    247: "active_site",   # R248 — DNA minor groove
    272: "active_site",   # R273 — DNA backbone
    275: "binding_site",  # A276 — DNA backbone
    276: "active_site",   # C277 — DNA major groove
    279: "active_site",   # R280 — DNA major groove
    # TET domain structural
    333: "structural",    # G334 — hinge
    336: "structural",    # R337 — salt bridge
    # Nuclear localization signal 1 (IDR/TET boundary)
    315: "signalling",    # P316 — NLS1
    316: "signalling",    # Q317 — NLS1
    317: "signalling",    # P318 — NLS1
    318: "signalling",    # K319 — NLS1
    319: "signalling",    # K320 — NLS1
    320: "signalling",    # K321 — NLS1
    321: "signalling",    # P322 — NLS1
    # Nuclear localization signal 2 (CTD)
    368: "signalling",    # L369 — NLS2
    369: "signalling",    # K370 — NLS2
    370: "signalling",    # S371 — NLS2
    371: "signalling",    # K372 — NLS2
    372: "signalling",    # K373 — NLS2
    373: "signalling",    # G374 — NLS2
    374: "signalling",    # Q375 — NLS2
    # Nuclear localization signal 3 (CTD)
    378: "signalling",    # R379 — NLS3
    379: "signalling",    # H380 — NLS3
    380: "signalling",    # K381 — NLS3
    381: "signalling",    # K382 — NLS3
    382: "signalling",    # L383 — NLS3
    383: "signalling",    # M384 — NLS3
}
# Filter out any positions that are out of range
_P53_ANNOTATIONS = {k: v for k, v in _P53_ANNOTATIONS.items()
                    if 0 <= k < len(_P53_SEQ)}

_P53_SS_REGIONS = [  # domain-level regions (no resolved SS; 0-indexed inclusive)
    (0,   39),   # TAD1   (positions 1–40)
    (40,  60),   # TAD2   (positions 41–61)
    (61,  91),   # PRD    (positions 62–92)
    (95, 291),   # DBD    (positions 96–292)
    (292, 322),  # IDR    (positions 293–323)
    (323, 355),  # TET    (positions 324–356)
    (363, 392),  # CTD    (positions 364–393)
]

P53 = ProteinDefinition(
    name="p53",
    uniprot_id="P04637",
    sequence=_P53_SEQ,
    annotations=_P53_ANNOTATIONS,
    category_weights=_P53_WEIGHTS.copy(),
    ss_regions=_P53_SS_REGIONS,
    default_n_variants=10000,  # ~75% single-mutant coverage (space=7,467)
)

# ---------------------------------------------------------------------------
# GB1 domain  (P06654, 56 aa)
# Key sites: Fab-binding (K10,T11,L12,K13,G14,E15,T16,T17,Y33,N35,D36,N37,G38),
#            Fc-binding (E27,K28,K31,Q32,D40,E42,W43),
#            Core packing (Y3,L5,L7,G9,T18,A26,F30,A34,V39,Y45,F52,V54),
#            Beta-hairpin folding (G41,T44,D46,D47,A48,T49,K50,T51,T53,T55,E56)
# Secondary structure: Beta Strands 1-4, Helix 1, IDR loops
# ---------------------------------------------------------------------------
_GB1_SEQ = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # 56 aa

_GB1_ANNOTATIONS: dict[int, str] = {
    # Core packing (structural)
    2:  "structural",   # Y3
    4:  "structural",   # L5
    6:  "structural",   # L7
    8:  "structural",   # G9
    17: "structural",   # T18
    25: "structural",   # A26
    29: "structural",   # F30
    33: "structural",   # A34
    38: "structural",   # V39
    44: "structural",   # Y45 (also beta hairpin, packing takes priority)
    51: "structural",   # F52
    53: "structural",   # V54
    # Fab / Fc binding interface
    9:  "binding_site", # K10 — Fab
    10: "binding_site", # T11 — Fab
    11: "binding_site", # L12 — Fab + packing
    12: "binding_site", # K13 — Fab
    13: "binding_site", # G14 — Fab
    14: "binding_site", # E15 — Fab
    15: "binding_site", # T16 — Fab + packing
    16: "binding_site", # T17 — Fab
    26: "binding_site", # E27 — most critical Fc
    27: "binding_site", # K28 — important Fc
    30: "binding_site", # K31 — important Fc
    31: "binding_site", # Q32 — important Fc
    32: "binding_site", # Y33 — Fab + packing
    34: "binding_site", # N35 — less important Fc + Fab
    35: "binding_site", # D36 — Fab
    36: "binding_site", # N37 — Fab
    37: "binding_site", # G38 — Fab
    39: "binding_site", # D40 — less important Fc
    41: "binding_site", # E42 — less important Fc + beta hairpin
    42: "binding_site", # W43 — less important Fc + packing + beta hairpin
    # Beta-hairpin folding
    40: "folding",      # G41 — beta hairpin
    43: "folding",      # T44 — beta hairpin
    45: "folding",      # D46 — beta hairpin
    46: "folding",      # D47 — beta hairpin
    47: "folding",      # A48 — beta hairpin
    48: "folding",      # T49 — beta hairpin
    49: "folding",      # K50 — beta hairpin
    50: "folding",      # T51 — beta hairpin
    52: "folding",      # T53 — beta hairpin
    54: "folding",      # T55 — beta hairpin
    55: "folding",      # E56 — beta hairpin
}

_GB1_SS_REGIONS = [           # 0-indexed from spreadsheet (1-indexed positions - 1)
    (0,  8),    # Beta Strand 1  (positions 1–9)
    (9,  10),   # IDR loop 1     (positions 10–11)
    (11, 19),   # Beta Strand 2  (positions 12–20)
    (20, 21),   # IDR loop 2     (positions 21–22)
    (22, 35),   # Helix 1        (positions 23–36)
    (36, 40),   # IDR loop 3     (positions 37–41)
    (41, 45),   # Beta Strand 3  (positions 42–46)
    (46, 48),   # IDR loop 4     (positions 47–49)
    (49, 55),   # Beta Strand 4  (positions 50–56)
]

GB1 = ProteinDefinition(
    name="GB1",
    uniprot_id="P06654",
    sequence=_GB1_SEQ,
    annotations={k: v for k, v in _GB1_ANNOTATIONS.items() if 0 <= k < len(_GB1_SEQ)},
    category_weights=_GB1_WEIGHTS.copy(),
    ss_regions=_GB1_SS_REGIONS,
    default_n_variants=1500,  # use exhaustive_singles=True to cover all 1,064 singles
)

PROTEINS: dict[str, ProteinDefinition] = {
    "T4Lysozyme": T4_LYSOZYME,
    "p53":        P53,
    "GB1":        GB1,
}


# ---------------------------------------------------------------------------
# Variant generator
# ---------------------------------------------------------------------------

def _mutate_sequence(
    sequence: str,
    n_mutations: int,
    rng: np.random.Generator,
) -> tuple[str, list[tuple[int, str, str]]]:
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


def _sample_n_mutations(rng: np.random.Generator, n_variants: int) -> list[int]:
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


def _ss_region_of(protein: ProteinDefinition, pos: int) -> int | None:
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
    mutations: list[tuple[int, str, str]],
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
    mutations: list[tuple[int, str, str]],
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
    mutations: list[tuple[int, str, str]],
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


class SyntheticDatasetGenerator:
    """
    Generates synthetic protein fitness datasets for IGLOO validation.

    Parameters
    ----------
    protein_name : str
        One of 'T4Lysozyme', 'p53', 'GB1'.

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
        self._generated: dict[str, dict[float, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str = "additive",
        n_variants: int | None = None,
        seed: int = 42,
        exhaustive_singles: bool = False,
    ) -> pd.DataFrame:
        """
        Generate a synthetic dataset (noiseless fitness values).

        Noise is intentionally omitted here — add it at the model-training
        level so the ground-truth signal is always available in the CSV.

        Parameters
        ----------
        model               : 'additive' | 'epistatic' | 'nonlinear'
        n_variants          : number of sequences to generate (WT always included);
                              defaults to protein.default_n_variants when None
        seed                : random seed for reproducibility
        exhaustive_singles  : if True, enumerate ALL L×19 single mutants before
                              filling remaining budget with random doubles/triples;
                              recommended for small proteins (e.g. GB1, 56 aa)

        Returns
        -------
        pd.DataFrame with columns:
            SEQ, fitness, n_mutations, mutation_string, model, protein, variant_type
        """
        if model not in _FITNESS_MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from {list(_FITNESS_MODELS.keys())}")

        n_variants = n_variants if n_variants is not None else self.protein.default_n_variants
        df = self._generate_one(model=model, n_variants=n_variants, seed=seed,
                                exhaustive_singles=exhaustive_singles)

        # Cache
        self._generated[model] = df

        return df

    def generate_all(
        self,
        n_variants: int | None = None,
        seed: int = 42,
        exhaustive_singles: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate all three dataset types (noiseless).

        Parameters
        ----------
        n_variants         : defaults to protein.default_n_variants when None
        exhaustive_singles : enumerate all L×19 single mutants (recommended for
                             small proteins like GB1)

        Returns dict: model_name → DataFrame.
        """
        n_variants = n_variants if n_variants is not None else self.protein.default_n_variants
        return {
            model: self._generate_one(model=model, n_variants=n_variants, seed=seed,
                                      exhaustive_singles=exhaustive_singles)
            for model in _FITNESS_MODELS
        }

    def save_all(
        self,
        output_dir: str = "data/synthetic",
        n_variants: int | None = None,
        seed: int = 42,
        exhaustive_singles: bool = False,
    ) -> list[str]:
        """
        Generate all datasets (noiseless) and save as CSVs + ground-truth JSON.

        File naming convention:
            {protein}_{model}.csv
            {protein}_ground_truth_weights.json

        Parameters
        ----------
        n_variants         : defaults to protein.default_n_variants when None
        exhaustive_singles : enumerate all L×19 single mutants before filling
                             remaining budget with random doubles/triples

        Returns list of saved file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved   = []

        all_data = self.generate_all(n_variants=n_variants, seed=seed,
                                     exhaustive_singles=exhaustive_singles)

        for model, df in all_data.items():
            fname = out_dir / f"{self.protein.name}_{model}.csv"
            df.to_csv(fname, index=False)
            print(f"  Saved {fname}  ({len(df)} rows)")
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
        seed: int,
        exhaustive_singles: bool = False,
    ) -> pd.DataFrame:
        rng    = np.random.default_rng(seed)
        wt_seq = self.protein.sequence
        fit_fn = _FITNESS_MODELS[model]

        rows      = []
        seen_seqs = {wt_seq}

        # Always include wild-type
        rows.append(self._wt_row(wt_seq, model))

        def _append_mutant(mutations: list[tuple[int, str, str]]) -> None:
            seq = list(wt_seq)
            for pos, _, mut_aa in mutations:
                seq[pos] = mut_aa
            seq_str = "".join(seq)
            seen_seqs.add(seq_str)
            fitness     = fit_fn(seq_str, self.protein, mutations)
            mut_str     = ";".join(f"{wt}{pos+1}{mut}" for pos, wt, mut in mutations)
            n_mut       = len(mutations)
            variant_type = {1: "single", 2: "double", 3: "triple"}.get(n_mut, "multi")
            rows.append({
                "SEQ":             seq_str,
                "fitness":         round(fitness, 6),
                "n_mutations":     n_mut,
                "mutation_string": mut_str,
                "model":           model,
                "protein":         self.protein.name,
                "variant_type":    variant_type,
                "is_wildtype":     False,
            })

        if exhaustive_singles:
            # Enumerate every possible single-point mutant (L × 19)
            for pos in range(len(wt_seq)):
                wt_aa = wt_seq[pos]
                for mut_aa in AMINO_ACIDS:
                    if mut_aa == wt_aa:
                        continue
                    _append_mutant([(pos, wt_aa, mut_aa)])
            # Fill remaining budget with random doubles/triples
            n_extra = n_variants - len(wt_seq) * (len(AMINO_ACIDS) - 1)
            if n_extra > 0:
                extra_counts = rng.choice([2, 3], size=n_extra, p=[2/3, 1/3])
                for n_mut in extra_counts:
                    for _ in range(10):
                        seq, mutations = _mutate_sequence(wt_seq, n_mut, rng)
                        if seq not in seen_seqs:
                            break
                    _append_mutant(mutations)
        else:
            mut_counts = _sample_n_mutations(rng, n_variants)
            for n_mut in mut_counts:
                # Retry up to 10 times to get a unique sequence
                for _ in range(10):
                    seq, mutations = _mutate_sequence(wt_seq, n_mut, rng)
                    if seq not in seen_seqs:
                        break
                _append_mutant(mutations)

        return pd.DataFrame(rows)

    def _wt_row(self, wt_seq: str, model: str) -> dict:
        return {
            "SEQ":             wt_seq,
            "fitness":         1.0,
            "n_mutations":     0,
            "mutation_string": "WT",
            "model":           model,
            "protein":         self.protein.name,
            "variant_type":    "wildtype",
            "is_wildtype":     True,
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

def split_train_test(
    df: pd.DataFrame,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a synthetic fitness DataFrame into train+val and held-out test subsets.

    - WT/wildtype sequence always goes to train+val.
    - Non-wildtype variants are stratified by *variant_type*
      ('single', 'double', 'triple') so the test set preserves the
      same mutation-order composition as the full dataset.
    - Any remaining variant types are placed entirely in train+val.

    Parameters
    ----------
    df            : DataFrame produced by SyntheticDatasetGenerator.generate()
    test_fraction : fraction of non-wildtype variants to hold out (default 0.20)
    seed          : random seed for reproducibility

    Returns
    -------
    train_val_df, test_df
        Both DataFrames are reset-indexed.  WT is always in train_val_df.
    """
    rng = np.random.default_rng(seed)

    # Wildtype rows always go to train+val
    wt_col = "is_wildtype" if "is_wildtype" in df.columns else None
    if wt_col:
        wt_mask = df[wt_col].astype(bool)
    else:
        wt_mask = df.get("variant_type", pd.Series(dtype=str)) == "wildtype"

    wt_df     = df[wt_mask].copy()
    non_wt_df = df[~wt_mask].copy()

    train_val_parts: list[pd.DataFrame] = [wt_df]
    test_parts:      list[pd.DataFrame] = []

    # Stratify over known mutation-order types
    for vtype in ("single", "double", "triple"):
        group = non_wt_df[non_wt_df["variant_type"] == vtype]
        if group.empty:
            continue
        idx = group.index.tolist()
        shuffled = rng.permutation(idx).tolist()
        n_test = max(1, round(len(shuffled) * test_fraction))
        test_parts.append(group.loc[shuffled[:n_test]])
        train_val_parts.append(group.loc[shuffled[n_test:]])

    # Any other variant types (e.g. 'multi') go entirely to train+val
    known_types = {"single", "double", "triple"}
    extra = non_wt_df[~non_wt_df["variant_type"].isin(known_types)]
    if not extra.empty:
        train_val_parts.append(extra)

    train_val_df = pd.concat(train_val_parts, ignore_index=True)
    test_df      = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=df.columns)
    )

    # Shuffle train+val rows
    train_val_df = train_val_df.sample(
        frac=1, random_state=int(rng.integers(0, 2**31))
    ).reset_index(drop=True)

    return train_val_df, test_df


def generate_all_proteins(
    output_dir: str = "data/synthetic",
    n_variants: int | None = None,
    seed: int = 42,
    exhaustive_singles: bool = False,
) -> dict[str, list[str]]:
    """
    Generate all three dataset types for all three proteins and save to disk.

    Fitness values are stored noiseless; add noise at the model-training level.

    Parameters
    ----------
    n_variants         : variants per protein; when None each protein uses its
                         own protein.default_n_variants (GB1=1500, T4=4000, p53=10000)
    exhaustive_singles : if True, enumerate all L×19 single mutants for every
                         protein before filling the remaining budget with
                         random doubles/triples (guarantees 100% single coverage
                         for small proteins like GB1)

    Returns dict: protein_name → list of saved file paths.
    """
    results = {}
    for name in PROTEINS:
        print(f"\n{'='*60}")
        print(f"Generating synthetic datasets for {name}")
        print(f"{'='*60}")
        gen  = SyntheticDatasetGenerator(name)
        nv   = n_variants if n_variants is not None else gen.protein.default_n_variants
        saved = gen.save_all(output_dir=output_dir, n_variants=nv,
                             seed=seed, exhaustive_singles=exhaustive_singles)
        results[name] = saved
    return results
