"""
Wordsearch QA — direction-MCQ rewrite (R5 B4B, 2026-05-05).

Old design (multi-line "WORD direction @ (col, row)") was too hard
(R3 0.225); custom verifier silently rejected valid responses, and
multi-line output exhausted the token budget.

New design: ask "What direction is the word X in the grid?" with
5-MCQ:
  A. horizontal (left-to-right or right-to-left)
  B. vertical (top-to-bottom or bottom-to-top)
  C. diagonal going down-right or down-left
  D. diagonal going up-right or up-left
  E. Not in grid

Per-level config:
  L0/L1: 4x4 grid, 3-letter target word, ALL filler letters NOT in target.
  L2-L4: 5x5 grid, 4-letter target word, some filler may match prefixes.
  L5-L7: 6x6 grid, 5-letter target word, multiple decoy partial matches.
  L8/L9: 8x8 grid + 30-40% trap (target word NOT placed; answer = E).

Verifier: standalone_base built-in (single MCQ letter, parent _check_answer).
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# Direction categories for MCQ
# A: horizontal (right or left)
# B: vertical (down or up)
# C: down-right or down-left (going downward diagonal)
# D: up-right or up-left (going upward diagonal)
# E: Not in grid
_DIR_TO_CATEGORY = {
    "right":      "A",  # horizontal
    "left":       "A",  # horizontal
    "down":       "B",  # vertical
    "up":         "B",  # vertical
    "down_right": "C",  # diagonal-down
    "down_left":  "C",  # diagonal-down
    "up_right":   "D",  # diagonal-up
    "up_left":    "D",  # diagonal-up
}

_DIRS = {
    "right":      (1, 0),
    "left":       (-1, 0),
    "down":       (0, 1),
    "up":         (0, -1),
    "down_right": (1, 1),
    "down_left":  (-1, 1),
    "up_right":   (1, -1),
    "up_left":    (-1, -1),
}


# Word bank by length.
_WORD_BANK = {
    3: ["CAT", "DOG", "SUN", "ICE", "RED", "TOP", "BOX", "OWL", "HEN",
        "BEE", "FOX", "ANT", "BUS", "JET", "PEN", "MAP", "EYE", "EAR",
        "EGG", "FAN", "OAK"],
    4: ["TREE", "FISH", "MOON", "STAR", "ROSE", "WIND", "RAIN", "BIRD",
        "LION", "WOLF", "FROG", "GOLD", "BLUE", "MILK", "BOOK", "DESK",
        "LAMP", "DOOR", "CAKE", "RING", "DUCK"],
    5: ["APPLE", "BREAD", "CLOUD", "EARTH", "FLAME", "GRASS", "HORSE",
        "HOUSE", "MOUSE", "OCEAN", "PEACH", "QUEEN", "RIVER", "SNAKE",
        "TIGER", "WATER", "ZEBRA", "MUSIC", "PIANO", "STONE"],
    6: ["BANANA", "CASTLE", "DOLLAR", "FLOWER", "GARDEN",
        "ISLAND", "MARKET", "NUMBER", "PLANET", "RABBIT", "SILVER",
        "TURTLE", "WINDOW", "YELLOW", "FOREST", "MEMORY", "ORANGE"],
}


def _can_place(grid, word, col, row, dx, dy):
    n = len(grid)
    for k, ch in enumerate(word):
        c = col + dx * k
        r = row + dy * k
        if c < 0 or c >= n or r < 0 or r >= n:
            return False
        if grid[r][c] is not None and grid[r][c] != ch:
            return False
    return True


def _place(grid, word, col, row, dx, dy):
    for k, ch in enumerate(word):
        grid[row + dy * k][col + dx * k] = ch


def _try_place(grid, word, rng, allowed_dirs):
    """Try random placements; return (col, row, dir_name) or None."""
    n = len(grid)
    dirs = list(allowed_dirs)
    rng.shuffle(dirs)
    for d_name in dirs:
        dx, dy = _DIRS[d_name]
        positions = [(c, r) for r in range(n) for c in range(n)]
        rng.shuffle(positions)
        for c, r in positions:
            if _can_place(grid, word, c, r, dx, dy):
                _place(grid, word, c, r, dx, dy)
                return (c, r, d_name)
    return None


def _word_present_anywhere(grid, word):
    """Return True if word appears in the grid in any of the 8 directions."""
    n = len(grid)
    for dx, dy in _DIRS.values():
        for r in range(n):
            for c in range(n):
                ok = True
                for k, ch in enumerate(word):
                    cc, rr = c + dx * k, r + dy * k
                    if cc < 0 or cc >= n or rr < 0 or rr >= n:
                        ok = False; break
                    if grid[rr][cc] != ch:
                        ok = False; break
                if ok:
                    return True
    return False


_QUESTION_STEMS = [
    "What direction is the word \"{word}\" in the grid below?",
    "Find the word \"{word}\" in the letter grid. Which direction is it placed?",
    "Locate the word \"{word}\" in the grid. What direction does it run?",
    "In the letter grid below, what direction is \"{word}\" placed?",
    "The word \"{word}\" is hidden in the grid (or it might not be present). Identify its direction.",
]


class WordSearchQA(StandaloneVisualEnv):
    """Direction-MCQ wordsearch puzzle. Pure single-letter MCQ A-E."""

    # 2026-05-05 R5 B4B: orientation-sensitive (rotate would flip horizontal
    # ↔ vertical and corrupt the answer category).
    ALLOW_ROTATION = False

    ENV_NAME = "wordsearch"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 1:
            # L0/L1: 4x4 grid, 3-letter word, only horizontal/vertical, no trap.
            return {
                "n": 4, "word_len": 3,
                "allowed_dirs": ["right", "down"],
                "filler_strategy": "non_target_letters",
                "trap_rate": 0.0,
            }
        if level <= 4:
            return {
                "n": 5, "word_len": 4,
                "allowed_dirs": ["right", "down", "up", "left"],
                "filler_strategy": "any",
                "trap_rate": 0.0,
            }
        if level <= 7:
            return {
                "n": 6, "word_len": 5,
                "allowed_dirs": ["right", "down", "up", "left",
                                 "down_right", "down_left",
                                 "up_right", "up_left"],
                "filler_strategy": "any",
                "trap_rate": 0.0,
            }
        # L8/L9: 8x8 grid, 5-letter word, all 8 directions, 30-40% trap
        return {
            "n": 8, "word_len": 5,
            "allowed_dirs": list(_DIRS.keys()),
            "filler_strategy": "any",
            "trap_rate": 0.30 if level == 8 else 0.40,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1601 + level * 89 + 53)

        for _ in range(40):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n"]
        word_len = cfg["word_len"]
        word_pool = list(_WORD_BANK.get(word_len, []))
        rng.shuffle(word_pool)
        if not word_pool:
            return None

        grid: List[List[Optional[str]]] = [[None] * n for _ in range(n)]
        target_word = word_pool[0]
        use_trap = (rng.random() < cfg["trap_rate"])
        true_direction_letter = None

        if use_trap:
            # Don't place the target word. Fill with random letters that
            # don't form the target.
            for _attempt in range(20):
                # Reset & fill
                grid = [[None] * n for _ in range(n)]
                for r in range(n):
                    for c in range(n):
                        if cfg["filler_strategy"] == "non_target_letters":
                            target_chars = set(target_word)
                            avail = [ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     if ch not in target_chars]
                            grid[r][c] = rng.choice(avail) if avail else "Z"
                        else:
                            grid[r][c] = chr(rng.randint(ord("A"), ord("Z")))
                if not _word_present_anywhere(
                    [["" if c is None else c for c in row] for row in grid],
                    target_word,
                ):
                    break
            else:
                return None
            true_direction_letter = "E"
        else:
            # Place the word in a random allowed direction.
            res = _try_place(grid, target_word, rng, cfg["allowed_dirs"])
            if res is None:
                return None
            _c, _r, d_name = res
            true_direction_letter = _DIR_TO_CATEGORY[d_name]
            # Fill remaining cells.
            for r in range(n):
                for c in range(n):
                    if grid[r][c] is None:
                        if cfg["filler_strategy"] == "non_target_letters":
                            target_chars = set(target_word)
                            avail = [ch for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                     if ch not in target_chars]
                            grid[r][c] = rng.choice(avail) if avail else "Z"
                        else:
                            grid[r][c] = chr(rng.randint(ord("A"), ord("Z")))

        # Build canonical answer (single letter A-E)
        answer = true_direction_letter
        grid_str: List[List[str]] = [[str(c) for c in row] for row in grid]

        # Build prompt
        stem_idx = (self.seed or 0) % len(_QUESTION_STEMS)
        stem = _QUESTION_STEMS[stem_idx].format(word=target_word)
        opts_text = (
            "  A. Horizontal (left-to-right or right-to-left)\n"
            "  B. Vertical (top-to-bottom or bottom-to-top)\n"
            "  C. Diagonal going downward (down-right or down-left)\n"
            "  D. Diagonal going upward (up-right or up-left)\n"
            "  E. Not in grid"
        )
        question = (
            f"{stem}\n\n"
            f"{opts_text}\n\n"
            f"Reply with the single letter (A/B/C/D/E) inside <answer>...</answer>. "
            f"Example: <answer>A</answer>"
        )
        if level == 0:
            question += (
                f" Hint: scan rows for {target_word}; if found left-to-right or "
                f"right-to-left, answer A. Answer: {answer}."
            )
        elif level == 1:
            question += (
                f" Hint: scan rows (horizontal=A) and columns (vertical=B); "
                f"if not found, the word is on a diagonal or not present."
            )
        elif level <= 4:
            question += " Hint: scan rows (A), columns (B), and diagonals (C/D)."

        img = self._render(grid_str, target_word)
        return question, answer, img

    def _render(self, grid, target_word):
        n = len(grid)
        cell_size = 0.7
        fig_w = max(5.5, n * cell_size + 1.5)
        fig_h = max(5.5, n * cell_size + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Word at top
        ax.text(n / 2.0, n + 0.7, f"Find: {target_word}",
                ha="center", va="center",
                fontsize=14, fontweight="bold", color="#1a3a6e")
        # Grid cells + letters
        for r in range(n):
            for c in range(n):
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1,
                                           facecolor="#fdfdfd",
                                           edgecolor="#888888",
                                           linewidth=1.0))
                ax.text(c + 0.5, n - 1 - r + 0.5, grid[r][c],
                        ha="center", va="center", fontsize=22,
                        fontweight="bold", family="monospace",
                        color="#222222")
        ax.set_xlim(-0.5, n + 0.2)
        ax.set_ylim(-0.5, n + 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
