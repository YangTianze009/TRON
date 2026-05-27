"""
Labelled-edge jigsaw piece matching QA.

Show 4-6 small puzzle pieces. Each piece has letters (A, B, C, ...) printed
on its 4 sides (each side gets a label, possibly blank). The matching rule:
two pieces "fit together" if and only if they share an edge with the SAME
label letter. The model must select the option (a pair like "1+3" or "2+5")
that identifies the two pieces that fit together.

Difficulty levels 0-9:
  L0-L1: 4 pieces, only 1 matching pair, very few labelled edges.
  L2-L3: 4 pieces, 1 match, more labelled edges.
  L4-L5: 5 pieces, 1 match, more distractor labels.
  L6-L7: 5 pieces, 1 match, several near-misses (labels close but no match).
  L8-L9: 6 pieces, 1 match, dense labels.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_QUESTION_TEMPLATES = [
    "Each puzzle piece has letters on its sides. Two pieces fit together when they share a side with the SAME letter. Which option (A-D) names the two pieces that fit?",
    "Look at the labelled edges of each piece. Which pair of pieces (option A, B, C, or D) shares a matching edge label?",
    "Identify the pair of pieces whose touching edges have matching letters. Choose A, B, C, or D.",
    "A side marked X must touch a side marked X. Which option (A, B, C, D) identifies the matching pair?",
    "Pick the option that names two pieces that can fit together by their labelled edges. Single letter A-D.",
    "Which pair of pieces has a matching edge label? Answer with the option letter (A, B, C, or D).",
    "Examine the edge labels carefully. Which pair (A, B, C, D) shares a matching letter?",
    "Two pieces fit when they share an edge label. Which option (A-D) is the correct matching pair?",
    "Determine which pair of pieces fits according to their edge labels. Pick A, B, C, or D.",
    "Which option (A, B, C, D) identifies two pieces that share a labelled edge?",
    "Choose the correct pair of pieces by matching edge labels. Single letter answer (A-D).",
    "After comparing all pieces, which option (A, B, C, D) gives the matching pair of pieces?",
    "Inspect each piece's edge labels. Which option names the pair that fits together (A, B, C, D)?",
    "Find the pair of pieces with a shared edge label. Answer A, B, C, or D.",
    "Which option (A through D) identifies the two pieces whose adjacent edges have the same label?",
    "Pick the correctly matching pair of pieces from the options A-D. Single letter only.",
]


def _draw_piece(ax, x: float, y: float, size: float, edge_labels: Dict[str, str],
                fill: str = "#fff8e1", edge: str = "#222"):
    """Draw a square piece centered at (x, y) with side labels.

    edge_labels = {"top": "A", "right": "", "bottom": "C", "left": ""}
    Empty string means no label.
    """
    s = size / 2
    ax.add_patch(mpatches.Rectangle((x - s, y - s), 2 * s, 2 * s,
                                     facecolor=fill, edgecolor=edge,
                                     linewidth=1.6, zorder=2))
    # Labels just inside each edge
    text_kw = dict(fontsize=15, fontweight="bold", color="#222",
                   ha="center", va="center", zorder=4)
    if edge_labels.get("top"):
        ax.text(x, y + s - s * 0.18, edge_labels["top"], **text_kw)
    if edge_labels.get("bottom"):
        ax.text(x, y - s + s * 0.18, edge_labels["bottom"], **text_kw)
    if edge_labels.get("left"):
        ax.text(x - s + s * 0.18, y, edge_labels["left"], **text_kw)
    if edge_labels.get("right"):
        ax.text(x + s - s * 0.18, y, edge_labels["right"], **text_kw)


class JigsawLabelledEdgeQA(StandaloneVisualEnv):
    ENV_NAME = "jigsaw_labelled_edge"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Labels needed: 1 (matching label, shared by 2 pieces) + 2*extra
        # (each matching piece's personal extras) + (n_pieces - 2)*(1 + extra)
        # (non-matching pieces' base + extras). Set n_labels accordingly.
        if level <= 1:
            n_pieces, extra = 4, 0
        elif level <= 3:
            n_pieces, extra = 4, 1
        elif level <= 5:
            n_pieces, extra = 5, 1
        elif level <= 7:
            n_pieces, extra = 5, 2
        else:
            n_pieces, extra = 6, 2
        n_labels = 1 + 2 * extra + (n_pieces - 2) * (1 + extra) + 1
        # Cap labels at 26 (A-Z)
        n_labels = min(n_labels, 26)
        return {"n_pieces": n_pieces, "n_labels": n_labels,
                "extra_labels_per_piece": extra}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(int(parameter.get("level", 0)), 9))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2401)

        for _ in range(30):
            result = self._try_generate(sub_rng, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict
                      ) -> Optional[Tuple[str, str, Image.Image]]:
        n_pieces = cfg["n_pieces"]
        n_labels = cfg["n_labels"]
        extra = cfg["extra_labels_per_piece"]

        labels = [chr(ord("A") + i) for i in range(n_labels)]
        # Sanity: we need >= 1 (match_label) + (n_pieces - 2) * (1 + extra)
        # + 2 * extra distinct labels in pool.
        needed = 1 + (n_pieces - 2) * (1 + extra) + 2 * extra
        if needed > n_labels:
            return None

        # Pick exactly one matching pair: pieces (i, j) share a label X.
        match_i, match_j = rng.sample(range(n_pieces), 2)
        match_label = rng.choice(labels)
        remaining = [l for l in labels if l != match_label]
        rng.shuffle(remaining)

        sides = ["top", "right", "bottom", "left"]
        pieces = [{s: "" for s in sides} for _ in range(n_pieces)]

        # Assign matching label to one side of each matching piece.
        side_i, side_j = rng.sample(sides, 2)
        pieces[match_i][side_i] = match_label
        pieces[match_j][side_j] = match_label

        # Each piece must have a unique label (so no piece is blank). Also,
        # every label other than match_label must be claimed by at most ONE
        # piece (else there'd be another matching pair).
        cursor = 0
        for idx in range(n_pieces):
            # Decide how many extra labels this piece gets
            # Matching pieces already have one label; non-matching get one as
            # base, then optionally `extra` more.
            n_more = extra
            if idx not in (match_i, match_j):
                # Non-matching piece needs at least 1 label (base) plus extra
                n_more = 1 + extra
            for _ in range(n_more):
                if cursor >= len(remaining):
                    break
                lbl = remaining[cursor]
                cursor += 1
                # Pick a free side
                free_sides = [s for s in sides if not pieces[idx][s]]
                if not free_sides:
                    continue
                side = rng.choice(free_sides)
                pieces[idx][side] = lbl

        # Verify: only (match_i, match_j) share a label.
        def _share_label(a, b):
            a_set = {v for v in a.values() if v}
            b_set = {v for v in b.values() if v}
            return bool(a_set & b_set)

        for a in range(n_pieces):
            for b in range(a + 1, n_pieces):
                if (a, b) == (min(match_i, match_j), max(match_i, match_j)):
                    continue
                if _share_label(pieces[a], pieces[b]):
                    return None

        # Sanity: matching pair must actually share match_label
        if not _share_label(pieces[match_i], pieces[match_j]):
            return None

        # Sanity: every piece has at least 1 label (no blank pieces)
        for p in pieces:
            if not any(p[s] for s in sides):
                return None

        # Build options: 4 pairs, one is correct
        correct_pair = tuple(sorted([match_i + 1, match_j + 1]))
        all_pairs = []
        for a in range(n_pieces):
            for b in range(a + 1, n_pieces):
                pair = (a + 1, b + 1)
                if pair == correct_pair:
                    continue
                all_pairs.append(pair)
        if len(all_pairs) < 3:
            return None
        rng.shuffle(all_pairs)
        distractors = all_pairs[:3]

        options = [correct_pair] + distractors
        rng.shuffle(options)
        correct_idx = options.index(correct_pair)
        answer_letter = chr(ord("A") + correct_idx)

        img = self._render(pieces, options, rng)
        sidx = (self.seed or 0) % len(_QUESTION_TEMPLATES)
        question = _QUESTION_TEMPLATES[sidx]
        return question, answer_letter, img

    def _render(self, pieces: List[Dict], options: List[Tuple[int, int]],
                rng) -> Image.Image:
        style = self._random_style()
        n = len(pieces)
        # Layout: pieces arranged in a row, options listed below
        fig_w = max(8.0, 1.6 * n)
        fig_h = 6.5
        fig, (ax_pieces, ax_opts) = plt.subplots(
            2, 1, figsize=(fig_w, fig_h),
            gridspec_kw={"height_ratios": [1.5, 1.0]})
        fig.patch.set_facecolor(style["bg_color"])

        ax_pieces.set_facecolor(style["bg_color"])
        ax_pieces.set_xlim(0, n * 1.4 + 0.2)
        ax_pieces.set_ylim(0, 1.6)
        ax_pieces.set_aspect("equal")
        ax_pieces.axis("off")
        ax_pieces.set_title("Puzzle pieces", fontsize=12, fontweight="bold")

        # Pick a light fill so dark edge-labels remain readable.
        light_fills = ["#fff8e1", "#f1f8e9", "#e3f2fd", "#fce4ec", "#fff3e0"]
        piece_fill = rng.choice(light_fills)
        for i, p in enumerate(pieces):
            cx = 0.7 + i * 1.4
            cy = 0.8
            _draw_piece(ax_pieces, cx, cy, 1.0, p, fill=piece_fill)
            ax_pieces.text(cx, 0.05, f"{i + 1}", fontsize=14,
                            fontweight="bold", ha="center", va="center",
                            color="#000")

        ax_opts.set_facecolor(style["bg_color"])
        ax_opts.set_xlim(0, 1)
        ax_opts.set_ylim(0, len(options) * 0.45)
        ax_opts.axis("off")
        ax_opts.set_title("Options", fontsize=11, fontweight="bold")
        for i, opt in enumerate(options):
            label = chr(ord("A") + i)
            y = (len(options) - 1 - i) * 0.45 + 0.1
            ax_opts.text(0.05, y, f"({label}) Pieces {opt[0]} and {opt[1]}",
                          fontsize=14, ha="left", va="center", color="#222")

        return self.fig_to_pil(fig, dpi=style["dpi"])
