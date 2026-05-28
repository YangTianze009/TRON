"""
Punnett Square QA environment.

Draws 2x2 (monohybrid) or 4x4 (dihybrid) Punnett squares with parent
genotypes and offspring cells. Questions about ratios, probabilities,
dominant/recessive fractions.
Targets: a multimodal benchmark science diagrams, MMMU biology.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# ------------------------------------------------------------------ #
# Genetics helpers
# ------------------------------------------------------------------ #

_TRAIT_NAMES_MONO = [
    ("Flower color", "Purple", "White", "C"),
    ("Seed shape", "Round", "Wrinkled", "R"),
    ("Plant height", "Tall", "Short", "T"),
    ("Seed color", "Yellow", "Green", "Y"),
    ("Pod shape", "Inflated", "Constricted", "I"),
    ("Eye color", "Brown", "Blue", "B"),
    ("Hair type", "Curly", "Straight", "H"),
    ("Fur color", "Black", "White", "F"),
]

_TRAIT_NAMES_DI = [
    (("Flower color", "Purple", "White", "C"),
     ("Seed shape", "Round", "Wrinkled", "R")),
    (("Seed color", "Yellow", "Green", "Y"),
     ("Plant height", "Tall", "Short", "T")),
    (("Eye color", "Brown", "Blue", "B"),
     ("Hair type", "Curly", "Straight", "H")),
]

_PHENOTYPE_COLORS = {
    "dominant": "#AED6F1",
    "recessive": "#F9E79F",
    "mixed": "#ABEBC6",
}

# Question phrasing templates for diversity.
_PROB_GENOTYPE_TEMPLATES = [
    "What is the probability of an offspring having genotype {g}? Express as a simplified fraction.",
    "From the Punnett square, what fraction of offspring have genotype {g}?",
    "Looking at the cross shown, what is the simplified probability of genotype {g}?",
    "Out of the 4 offspring cells, what simplified fraction has genotype {g}?",
]

_PROB_PHENOTYPE_TEMPLATES = [
    "What is the probability that an offspring shows the {p} ({w}) phenotype? Express as a simplified fraction.",
    "From the Punnett square, what simplified fraction of offspring display the {p} phenotype?",
    "What fraction of offspring will exhibit {p} ({w})? Give a simplified fraction.",
]

_OFFSPRING_RATIO_TEMPLATES_MONO = [
    "In the Punnett square for {trait}, what is the genotype ratio of offspring? Express as numbers separated by colons (e.g., 1:2:1), listing genotypes alphabetically.",
    "Count the offspring genotypes in alphabetical order. Report as a colon-separated ratio.",
]

_CARRIER_TEMPLATES = [
    "What fraction of offspring are carriers (heterozygous) for {trait}? Express as a simplified fraction.",
    "What simplified fraction of offspring are heterozygous ({trait} carriers)?",
]

_DOMINANT_FRACTION_TEMPLATES = [
    "What fraction of offspring will show the dominant phenotype ({p})? Express as a simplified fraction.",
    "What simplified fraction of offspring express the dominant trait ({p})?",
]

_DI_RATIO_TEMPLATES = [
    "What is the phenotype ratio of offspring in this dihybrid cross? Express as numbers separated by colons.",
    "List the phenotype counts for the dihybrid cross as a colon-separated ratio.",
]

_DI_DOM_TEMPLATES = [
    "What fraction of offspring show BOTH dominant phenotypes ({d1} and {d2})? Express as a simplified fraction.",
    "What simplified fraction of offspring express both {d1} and {d2}?",
]

_DI_SPECIFIC_TEMPLATES = [
    "Using the dihybrid Punnett square shown (16 cells), if we scaled up to 64 offspring, how many would have genotype '{g}'? (Count the occurrences of '{g}' in the grid, then multiply by 4.)",
    "Based on the 4x4 Punnett square in the image, count how many cells contain genotype '{g}'. Then, scaled to 64 offspring, how many would have that genotype?",
    "From the Punnett square shown, determine the frequency of genotype '{g}' in the 16 offspring cells, and report the expected count in a population of 64 offspring.",
]

def _genotype_to_phenotype(genotype: str, dom_allele: str) -> str:
    """Given a genotype like 'Cc', return 'dominant' or 'recessive'."""
    return "dominant" if dom_allele.upper() in genotype.upper() else "recessive"

def _make_gametes_mono(genotype: str) -> List[str]:
    """Return list of gametes from a monohybrid genotype like 'Cc'."""
    return [genotype[0], genotype[1]]

def _make_gametes_di(genotype: str) -> List[str]:
    """Return gametes for dihybrid, e.g. 'CcRr' -> ['CR','Cr','cR','cr']."""
    a1, a2 = genotype[0], genotype[1]
    b1, b2 = genotype[2], genotype[3]
    return [a1 + b1, a1 + b2, a2 + b1, a2 + b2]

class PunnettSquareQA(StandaloneVisualEnv):
    ENV_NAME = "punnett_square"

    QUESTION_TYPES = [
        "offspring_ratio", "probability_genotype", "probability_phenotype",
        "dominant_fraction", "carrier_fraction",
        # Harder type
        "dihybrid_specific_genotype",
    ]

    def _level_config(self, level: int) -> Dict:
        # Difficulty redesign 2026-04-17. Previous L3 dipped to 0.7 because
        # offspring_ratio requires a specific 1:2:1 / 3:1 format that the
        # model easily mis-phrases. Move offspring_ratio to mid-levels and
        # use cleaner mono crosses for L0-L3.
        if level <= 0:
            return {"qtypes": ["probability_genotype", "probability_phenotype"],
                    "qweights": [5, 5], "cross": "mono",
                    "force_heterozygous": True}
        if level <= 2:
            return {"qtypes": ["probability_genotype", "probability_phenotype",
                               "carrier_fraction"],
                    "qweights": [4, 3, 3], "cross": "mono"}
        if level <= 4:
            return {"qtypes": ["probability_phenotype", "carrier_fraction",
                               "dominant_fraction"],
                    "qweights": [3, 4, 3], "cross": "mono"}
        if level <= 5:
            return {"qtypes": ["offspring_ratio", "probability_phenotype"],
                    "qweights": [5, 5], "cross": "mixed"}
        if level <= 7:
            return {"qtypes": ["probability_genotype", "probability_phenotype",
                               "dihybrid_specific_genotype"],
                    "qweights": [3, 3, 4], "cross": "di"}
        if level <= 8:
            return {"qtypes": ["dihybrid_specific_genotype", "probability_genotype"],
                    "qweights": [6, 4], "cross": "di"}
        return {"qtypes": ["dihybrid_specific_genotype"],
                "qweights": [1], "cross": "di"}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Use sub_rng seeded by level so seed-variation is preserved per level.
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 3301)

        qtype = parameter.get("question_type")
        if qtype not in self.QUESTION_TYPES:
            qtype = sub_rng.choices(cfg["qtypes"], weights=cfg["qweights"], k=1)[0]

        # Force dihybrid for dihybrid_specific_genotype
        if qtype == "dihybrid_specific_genotype":
            cross_type = "di"
        elif cfg["cross"] == "mixed":
            cross_type = sub_rng.choice(["mono", "di"])
        else:
            cross_type = cfg["cross"]

        if cross_type == "mono":
            return self._generate_mono(sub_rng, qtype,
                                        force_het=cfg.get("force_heterozygous",
                                                         False))
        else:
            return self._generate_di(sub_rng, qtype)

    # ------------------------------------------------------------------ #
    # Monohybrid (2x2)
    # ------------------------------------------------------------------ #

    def _generate_mono(
        self, rng: random.Random, qtype: str, force_het: bool = False
    ) -> Optional[Tuple[str, str, Image.Image]]:
        trait_info = rng.choice(_TRAIT_NAMES_MONO)
        trait_name, dom_pheno, rec_pheno, letter = trait_info
        dom = letter.upper()
        rec = letter.lower()

        # Parent genotypes — pick from common crosses
        if force_het:
            # Het × het always gives interesting ratio 1:2:1 / 3:1.
            p1_geno, p2_geno = dom + rec, dom + rec
        else:
            crosses = [
                (dom + rec, dom + rec),  # Cc x Cc
                (dom + dom, rec + rec),  # CC x cc
                (dom + rec, rec + rec),  # Cc x cc
                (dom + dom, dom + rec),  # CC x Cc
            ]
            p1_geno, p2_geno = rng.choice(crosses)

        gametes_1 = _make_gametes_mono(p1_geno)
        gametes_2 = _make_gametes_mono(p2_geno)

        # Build offspring grid
        grid = []
        for g2 in gametes_2:
            row = []
            for g1 in gametes_1:
                # Normalize: uppercase first
                alleles = sorted([g1, g2], key=lambda x: x.lower())
                if alleles[0].isupper():
                    offspring = alleles[0] + alleles[1]
                elif alleles[1].isupper():
                    offspring = alleles[1] + alleles[0]
                else:
                    offspring = alleles[0] + alleles[1]
                row.append(offspring)
            grid.append(row)

        # Count genotypes
        all_offspring = [cell for row in grid for cell in row]
        geno_counts = {}
        for g in all_offspring:
            geno_counts[g] = geno_counts.get(g, 0) + 1

        total = len(all_offspring)
        dom_count = sum(1 for g in all_offspring if dom in g.upper()
                        and g.upper() != g.lower())
        # Actually: dominant phenotype = has at least one uppercase allele
        dom_pheno_count = sum(1 for g in all_offspring
                              if dom.upper() in g)
        rec_pheno_count = total - dom_pheno_count
        # Carriers: heterozygous = has both dom and rec allele
        carrier_count = sum(1 for g in all_offspring
                            if dom.upper() in g and rec.lower() in g)

        question, answer = self._make_qa_mono(
            rng, qtype, trait_name, dom_pheno, rec_pheno,
            dom, rec, all_offspring, geno_counts, total,
            dom_pheno_count, rec_pheno_count, carrier_count,
            p1_geno, p2_geno
        )
        if question is None:
            return None

        image = self._render_mono(
            rng, grid, gametes_1, gametes_2, p1_geno, p2_geno,
            trait_name, dom_pheno, rec_pheno, dom
        )
        return question, answer, image

    def _make_qa_mono(
        self, rng, qtype, trait_name, dom_pheno, rec_pheno,
        dom, rec, all_offspring, geno_counts, total,
        dom_pheno_count, rec_pheno_count, carrier_count,
        p1_geno, p2_geno
    ):
        if qtype == "offspring_ratio":
            # Genotype ratio
            unique = sorted(geno_counts.keys())
            parts = [f"{geno_counts[g]} {g}" for g in unique]
            ratio_str = " : ".join(str(geno_counts[g]) for g in unique)
            q = rng.choice(_OFFSPRING_RATIO_TEMPLATES_MONO).format(
                trait=trait_name)
            return q, ratio_str

        elif qtype == "probability_genotype":
            target = rng.choice(list(geno_counts.keys()))
            prob = geno_counts[target] / total
            q = rng.choice(_PROB_GENOTYPE_TEMPLATES).format(g=target)
            # Simplify
            from math import gcd
            g = gcd(geno_counts[target], total)
            num, den = geno_counts[target] // g, total // g
            return q, f"{num}/{den}"

        elif qtype == "probability_phenotype":
            which = rng.choice(["dominant", "recessive"])
            if which == "dominant":
                count = dom_pheno_count
                pheno = dom_pheno
            else:
                count = rec_pheno_count
                pheno = rec_pheno
            from math import gcd
            g = gcd(count, total)
            num, den = count // g, total // g
            q = rng.choice(_PROB_PHENOTYPE_TEMPLATES).format(p=pheno, w=which)
            return q, f"{num}/{den}"

        elif qtype == "dominant_fraction":
            from math import gcd
            g = gcd(dom_pheno_count, total)
            num, den = dom_pheno_count // g, total // g
            q = rng.choice(_DOMINANT_FRACTION_TEMPLATES).format(p=dom_pheno)
            return q, f"{num}/{den}"

        elif qtype == "carrier_fraction":
            from math import gcd
            if carrier_count == 0:
                q = (f"How many offspring out of {total} are carriers "
                     f"(heterozygous) for {trait_name}?")
                return q, "0"
            g = gcd(carrier_count, total)
            num, den = carrier_count // g, total // g
            q = rng.choice(_CARRIER_TEMPLATES).format(trait=trait_name)
            return q, f"{num}/{den}"

        return None, None

    # ------------------------------------------------------------------ #
    # Dihybrid (4x4)
    # ------------------------------------------------------------------ #

    def _generate_di(
        self, rng: random.Random, qtype: str
    ) -> Optional[Tuple[str, str, Image.Image]]:
        trait_pair = rng.choice(_TRAIT_NAMES_DI)
        t1, t2 = trait_pair
        trait1_name, dom1_pheno, rec1_pheno, letter1 = t1
        trait2_name, dom2_pheno, rec2_pheno, letter2 = t2

        dom1, rec1 = letter1.upper(), letter1.lower()
        dom2, rec2 = letter2.upper(), letter2.lower()

        # Both parents heterozygous for both traits (classic dihybrid)
        p1_geno = dom1 + rec1 + dom2 + rec2
        p2_geno = dom1 + rec1 + dom2 + rec2

        gametes_1 = _make_gametes_di(p1_geno)
        gametes_2 = _make_gametes_di(p2_geno)

        grid = []
        for g2 in gametes_2:
            row = []
            for g1 in gametes_1:
                # Combine: first trait alleles + second trait alleles
                a1 = sorted([g1[0], g2[0]], key=lambda x: (x.lower(), x))
                a2 = sorted([g1[1], g2[1]], key=lambda x: (x.lower(), x))
                # Uppercase first
                if a1[0].isupper():
                    part1 = a1[0] + a1[1]
                elif a1[1].isupper():
                    part1 = a1[1] + a1[0]
                else:
                    part1 = a1[0] + a1[1]
                if a2[0].isupper():
                    part2 = a2[0] + a2[1]
                elif a2[1].isupper():
                    part2 = a2[1] + a2[0]
                else:
                    part2 = a2[0] + a2[1]
                row.append(part1 + part2)
            grid.append(row)

        all_offspring = [cell for row in grid for cell in row]
        total = len(all_offspring)

        # Phenotype counts
        pheno_counts = {}
        for off in all_offspring:
            p1 = dom1_pheno if dom1 in off[:2] else rec1_pheno
            p2 = dom2_pheno if dom2 in off[2:] else rec2_pheno
            key = f"{p1}, {p2}"
            pheno_counts[key] = pheno_counts.get(key, 0) + 1

        geno_counts = {}
        for off in all_offspring:
            geno_counts[off] = geno_counts.get(off, 0) + 1

        # For dihybrid, redirect some question types
        if qtype == "carrier_fraction":
            qtype = "offspring_ratio"

        question, answer = self._make_qa_di(
            rng, qtype, trait1_name, trait2_name,
            dom1_pheno, rec1_pheno, dom2_pheno, rec2_pheno,
            dom1, dom2, all_offspring, geno_counts, pheno_counts, total
        )
        if question is None:
            return None

        image = self._render_di(
            rng, grid, gametes_1, gametes_2,
            trait1_name, trait2_name, dom1_pheno, rec1_pheno,
            dom2_pheno, rec2_pheno, dom1, dom2
        )
        return question, answer, image

    def _make_qa_di(
        self, rng, qtype, t1_name, t2_name,
        dom1_p, rec1_p, dom2_p, rec2_p,
        dom1, dom2, all_offspring, geno_counts, pheno_counts, total
    ):
        if qtype == "offspring_ratio":
            sorted_phenos = sorted(pheno_counts.keys())
            ratio_str = " : ".join(str(pheno_counts[p]) for p in sorted_phenos)
            q = rng.choice(_DI_RATIO_TEMPLATES)
            return q, ratio_str

        elif qtype == "probability_genotype":
            target = rng.choice(list(geno_counts.keys()))
            from math import gcd
            g = gcd(geno_counts[target], total)
            num, den = geno_counts[target] // g, total // g
            q = rng.choice(_PROB_GENOTYPE_TEMPLATES).format(g=target)
            return q, f"{num}/{den}"

        elif qtype == "probability_phenotype":
            target_pheno = rng.choice(list(pheno_counts.keys()))
            count = pheno_counts[target_pheno]
            from math import gcd
            g = gcd(count, total)
            num, den = count // g, total // g
            q = (f"What is the probability that an offspring shows "
                 f"'{target_pheno}'? Express as a simplified fraction.")
            return q, f"{num}/{den}"

        elif qtype == "dominant_fraction":
            # Both dominant
            both_dom_key = f"{dom1_p}, {dom2_p}"
            count = pheno_counts.get(both_dom_key, 0)
            from math import gcd
            g = gcd(count, total) if count > 0 else 1
            num, den = count // g, total // g
            q = rng.choice(_DI_DOM_TEMPLATES).format(d1=dom1_p, d2=dom2_p)
            return q, f"{num}/{den}"

        elif qtype == "dihybrid_specific_genotype":
            # Pick a specific genotype from the grid and ask its probability
            # Then ask: if we have 64 offspring, how many would have this genotype?
            target = rng.choice(list(geno_counts.keys()))
            count = geno_counts[target]
            expected_in_64 = count * 64 // total  # since total=16, this is count*4
            # Do NOT leak count in text; model must count from image.
            q = rng.choice(_DI_SPECIFIC_TEMPLATES).format(g=target)
            return q, str(expected_in_64)

        return None, None

    # ------------------------------------------------------------------ #
    # Rendering — Monohybrid 2x2
    # ------------------------------------------------------------------ #

    def _render_mono(
        self, rng, grid, gametes_1, gametes_2,
        p1_geno, p2_geno, trait_name, dom_pheno, rec_pheno, dom
    ) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        ax.set_xlim(-1.5, 3.5)
        ax.set_ylim(-3.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

        cell_size = 1.2
        x0, y0 = 0.0, 0.0  # top-left of grid

        # Title
        ax.text(1.2, 1.3, f"Punnett Square: {trait_name}",
                ha="center", fontsize=fs + 1, fontweight="bold", fontfamily=ff)
        ax.text(1.2, 1.0, f"Parent 1: {p1_geno}  x  Parent 2: {p2_geno}",
                ha="center", fontsize=fs - 2, fontfamily=ff)

        # Column headers (Parent 1 gametes)
        for j, g in enumerate(gametes_1):
            ax.text(x0 + j * cell_size + cell_size / 2, y0 + 0.3,
                    g, ha="center", va="center", fontsize=fs + 2, fontweight="bold",
                    color=palette[0], fontfamily=ff)

        # Row headers (Parent 2 gametes)
        for i, g in enumerate(gametes_2):
            ax.text(x0 - 0.4, y0 - i * cell_size - cell_size / 2,
                    g, ha="center", va="center", fontsize=fs + 2, fontweight="bold",
                    color=palette[1], fontfamily=ff)

        # Grid cells
        dom_color = palette[2]
        rec_color = palette[3]
        for i in range(2):
            for j in range(2):
                geno = grid[i][j]
                is_dom = dom.upper() in geno
                color = dom_color if is_dom else rec_color
                rect = mpatches.FancyBboxPatch(
                    (x0 + j * cell_size, y0 - (i + 1) * cell_size),
                    cell_size, cell_size,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor="black", linewidth=style["line_width"])
                ax.add_patch(rect)
                ax.text(x0 + j * cell_size + cell_size / 2,
                        y0 - i * cell_size - cell_size / 2,
                        geno, ha="center", va="center",
                        fontsize=fs + 4, fontweight="bold", fontfamily=ff)

        # Legend
        ax.add_patch(mpatches.Rectangle((x0, -3.2), 0.3, 0.3,
                     facecolor=dom_color, edgecolor="black"))
        ax.text(x0 + 0.4, -3.05, f"= {dom_pheno} (dominant)",
                fontsize=fs - 3, va="center", fontfamily=ff)
        ax.add_patch(mpatches.Rectangle((x0 + 1.5, -3.2), 0.3, 0.3,
                     facecolor=rec_color, edgecolor="black"))
        ax.text(x0 + 1.9, -3.05, f"= {rec_pheno} (recessive)",
                fontsize=fs - 3, va="center", fontfamily=ff)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

    # ------------------------------------------------------------------ #
    # Rendering — Dihybrid 4x4
    # ------------------------------------------------------------------ #

    def _render_di(
        self, rng, grid, gametes_1, gametes_2,
        t1_name, t2_name, dom1_p, rec1_p, dom2_p, rec2_p, dom1, dom2
    ) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(9 * sc, 9 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        palette = style["palette"]
        fs = style["font_size_base"]
        ff = style["font_family"]

        ax.set_xlim(-2.0, 6.5)
        ax.set_ylim(-6.5, 2.0)
        ax.set_aspect("equal")
        ax.axis("off")

        cell_size = 1.2
        x0, y0 = 0.0, 0.0

        ax.text(2.4, 1.6, f"Dihybrid Cross: {t1_name} & {t2_name}",
                ha="center", fontsize=fs + 1, fontweight="bold", fontfamily=ff)

        # Column headers
        for j, g in enumerate(gametes_1):
            ax.text(x0 + j * cell_size + cell_size / 2, y0 + 0.35,
                    g, ha="center", va="center", fontsize=fs, fontweight="bold",
                    color=palette[0], fontfamily=ff)

        # Row headers
        for i, g in enumerate(gametes_2):
            ax.text(x0 - 0.5, y0 - i * cell_size - cell_size / 2,
                    g, ha="center", va="center", fontsize=fs, fontweight="bold",
                    color=palette[1], fontfamily=ff)

        # Grid cells
        colors_map = {
            (True, True): palette[2],    # both dominant
            (True, False): palette[3],   # first dom, second rec
            (False, True): palette[4],   # first rec, second dom
            (False, False): palette[5],  # both recessive
        }
        for i in range(4):
            for j in range(4):
                geno = grid[i][j]
                is_dom1 = dom1.upper() in geno[:2]
                is_dom2 = dom2.upper() in geno[2:]
                color = colors_map[(is_dom1, is_dom2)]
                rect = mpatches.FancyBboxPatch(
                    (x0 + j * cell_size, y0 - (i + 1) * cell_size),
                    cell_size, cell_size,
                    boxstyle="round,pad=0.02",
                    facecolor=color, edgecolor="black", linewidth=style["line_width"])
                ax.add_patch(rect)
                ax.text(x0 + j * cell_size + cell_size / 2,
                        y0 - i * cell_size - cell_size / 2,
                        geno, ha="center", va="center",
                        fontsize=fs - 1, fontweight="bold", fontfamily=ff)

        # Legend
        ly = -5.5
        for idx, ((d1, d2), col) in enumerate(colors_map.items()):
            lx = x0 + idx * 1.6
            ax.add_patch(mpatches.Rectangle((lx, ly), 0.25, 0.25,
                         facecolor=col, edgecolor="black"))
            p1 = dom1_p if d1 else rec1_p
            p2 = dom2_p if d2 else rec2_p
            ax.text(lx + 0.35, ly + 0.12, f"{p1}, {p2}",
                    fontsize=fs - 4, va="center", fontfamily=ff)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])
