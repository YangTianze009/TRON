"""
Chiral Object Identification QA.

Shows two 2D projections of molecule-like ball-and-stick structures side by side.
Asks whether they are identical (rotations), mirror images, or different.

Difficulty axes:
  A) n_chiral_centers: 1..2
  B) color_similarity: distinct -> confusable
"""
import math, random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_DISTINCT_COLORS_POOL = [
    ['#e74c3c', '#2ecc71', '#3498db', '#f1c40f'],
    ['#1abc9c', '#9b59b6', '#34495e', '#f39c12'],
    ['#c0392b', '#27ae60', '#2980b9', '#d35400'],
    ['#16a085', '#8e44ad', '#2c3e50', '#e67e22'],
]
_CONFUSABLE_COLORS_POOL = [
    ['#e74c3c', '#e67e22', '#3498db', '#8e44ad'],   # red/orange + blue/purple
    ['#c0392b', '#d35400', '#2980b9', '#9b59b6'],   # darker
    ['#1abc9c', '#16a085', '#3498db', '#2980b9'],   # teal/blue
]

class ChiralObjectIdentificationQA(StandaloneVisualEnv):
    ENV_NAME = "chiral_object_identification"

    def _level_config(self, level):
        return {
            'n_centers': 1 + level // 4,
            'colors_pool': _DISTINCT_COLORS_POOL if level <= 4 else _CONFUSABLE_COLORS_POOL,
            # L0/L1: easy mode — molecules drawn at SAME perspective (no
            # confusing rotation) so identical/different is a direct color
            # match check rather than mental rotation.
            'easy_no_perspective': level <= 1,
        }

    def _draw_molecule(self, ax, center, substituents, colors, perspective=0):
        """Draw a tetrahedral center with 4 substituents in 2D."""
        cx, cy = center
        # 4 positions around center (tetrahedral projection)
        angles = [0, 90, 180, 270]
        offset = perspective * 15
        positions = []
        for i, a_deg in enumerate(angles):
            rad = math.radians(a_deg + offset)
            r = 1.2
            px, py = cx + r * math.cos(rad), cy + r * math.sin(rad)
            positions.append((px, py))

        for i, (px, py) in enumerate(positions):
            color = colors[substituents[i] % len(colors)]
            # Draw bond
            style = '-' if i < 2 else '--'  # wedge/dash notation simplified
            lw = 3 if i < 2 else 1.5
            ax.plot([cx, px], [cy, py], color='#333', linewidth=lw, linestyle=style)
            # Draw ball (identity encoded in color only — DO NOT label with
            # the substituent index, which would leak the answer).
            ax.plot(px, py, 'o', color=color, markersize=22,
                    markeredgecolor='#333', markeredgewidth=1.5)

        # Central atom
        ax.plot(cx, cy, 'o', color='#999', markersize=15, markeredgecolor='#333', markeredgewidth=2)

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1008)
        style = self._random_style()

        n_centers = cfg['n_centers']
        # Pick a fresh palette per seed for variety.
        colors = rng.choice(cfg['colors_pool'])

        # Generate first molecule
        mol1_subs = list(range(4))
        rng.shuffle(mol1_subs)

        # Decide relationship: identical (rotation), mirror, or different
        if cfg.get('easy_no_perspective'):
            # At L0/L1, restrict to obvious identical-vs-different cases.
            # 'identical' means truly identical (no rearrangement needed).
            # 'different' means at least 3 substituents differ in position.
            rel_type = rng.choice(['identical', 'different'])
        else:
            rel_type = rng.choice(['identical', 'mirror', 'different'])

        if rel_type == 'identical':
            if cfg.get('easy_no_perspective'):
                # Truly identical: same substituents in same positions
                mol2_subs = list(mol1_subs)
            else:
                # Rotate (cyclic permutation of 3 substituents)
                mol2_subs = list(mol1_subs)
                i, j = rng.sample(range(4), 2)
                mol2_subs[i], mol2_subs[j] = mol2_subs[j], mol2_subs[i]
                k = rng.choice([x for x in range(4) if x != i and x != j])
                mol2_subs[j], mol2_subs[k] = mol2_subs[k], mol2_subs[j]
            correct = 'A'
        elif rel_type == 'mirror':
            # Swap exactly 2 substituents (odd permutation)
            mol2_subs = list(mol1_subs)
            i, j = rng.sample(range(4), 2)
            mol2_subs[i], mol2_subs[j] = mol2_subs[j], mol2_subs[i]
            correct = 'B'
        else:
            if cfg.get('easy_no_perspective'):
                # At L0/L1, ensure mol2 differs from mol1 in at least 3 positions
                # so the "different" answer is visually obvious.
                while True:
                    mol2_subs = list(range(4))
                    rng.shuffle(mol2_subs)
                    n_diff = sum(1 for a, b in zip(mol1_subs, mol2_subs) if a != b)
                    if n_diff >= 3:
                        break
            else:
                mol2_subs = list(range(4))
                rng.shuffle(mol2_subs)
                # Ensure not same
                while mol2_subs == mol1_subs:
                    rng.shuffle(mol2_subs)
            correct = 'C'

        sc = style['figsize_scale']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10*sc, 5*sc))
        fig.patch.set_facecolor(style['bg_color'])
        for ax in (ax1, ax2):
            ax.set_facecolor(style['bg_color'])
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)

        # At L0/L1, lock perspective so identical/mirror is a simple visual
        # comparison (not a mental rotation puzzle).
        persp2 = 0 if cfg.get('easy_no_perspective') else rng.randint(0, 3)
        self._draw_molecule(ax1, (0, 0), mol1_subs, colors, perspective=0)
        self._draw_molecule(ax2, (0, 0), mol2_subs, colors, perspective=persp2)

        ax1.set_title("Structure I", fontsize=12, fontweight='bold')
        ax2.set_title("Structure II", fontsize=12, fontweight='bold')

        # Title pool
        title_pool = [
            "Chirality Comparison",
            "Stereochemistry Comparison",
            "Compare the Two Structures",
            "Molecular Comparison",
        ]
        fig.suptitle(rng.choice(title_pool),
                     fontsize=style['font_size_base'] + 3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        if cfg.get('easy_no_perspective'):
            templates = [
                ("Two molecular structures are shown side by side from the same "
                 "viewpoint. Compare the colored substituents at each position. "
                 "Are these structures:\n"
                 "(A) Identical (every substituent in the same position)\n"
                 "(B) Mirror images (enantiomers)\n"
                 "(C) Different molecules (substituents in different positions)\n"
                 "Answer with a single letter."),
                ("Examine the two structures shown. Both are drawn from the same "
                 "viewpoint. Pick the relationship between them:\n"
                 "(A) Identical\n(B) Mirror images (enantiomers)\n"
                 "(C) Different molecules\nAnswer with a single letter."),
            ]
        else:
            templates = [
                ("Two molecular structures are shown. Solid bonds point toward you; "
                 "dashed bonds point away. Are these structures:\n"
                 "(A) Identical (rotations of each other)\n"
                 "(B) Mirror images (enantiomers)\n"
                 "(C) Different molecules (neither rotation nor mirror)\n"
                 "Answer with a single letter."),
                ("Compare the two stereochemical structures. With solid bonds out "
                 "of the page and dashed bonds behind, decide:\n"
                 "(A) Identical (one is a rotation of the other)\n"
                 "(B) Enantiomers (mirror images)\n"
                 "(C) Distinct molecules\nAnswer with a single letter."),
                ("Are the depicted structures rotations, mirror-images, or "
                 "different molecules? Choose:\n(A) Identical (rotational copies)\n"
                 "(B) Mirror images (enantiomers)\n(C) Different molecules\n"
                 "Answer with a single letter."),
            ]
        q = rng.choice(templates)
        return q, correct, img

if __name__ == "__main__":
    env = ChiralObjectIdentificationQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
