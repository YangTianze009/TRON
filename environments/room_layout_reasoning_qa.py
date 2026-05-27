"""
Room Layout Reasoning QA.

Top-down floor plan with existing furniture. Candidate positions shown
as dashed outlines A/B/C/D. Question varies by level:

  L0-1:  "Which dashed position fits without overlapping existing furniture?"
  L2-3:  Either fit OR largest free area — pick the valid placement that meets
         a specified constraint (e.g. against a wall).
  L4-6:  "Which position places the new item adjacent to <existing piece X>?"
         or "Which placement keeps the walkway clear?" (a shaded corridor must
         remain free).
  L7-9:  L-shaped piece placement, multiple constraints (e.g. against wall AND
         adjacent to <X>), or "Which placement is invalid?" (negated query).

Primitives pool:
  rectangle furniture, L-shape furniture (at L>=5).
  walkway strips (at L>=4).
  4+ question templates.
Colors/names/shapes shuffled per seed.
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

_FURNITURE_POOL = ["Table", "Chair", "Shelf", "Desk", "Bed", "Sofa",
                   "Dresser", "Rug", "Lamp", "Plant"]

class RoomLayoutReasoningQA(StandaloneVisualEnv):
    ENV_NAME = "room_layout_reasoning"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        return {
            "level": level,
            "n_pieces":      2 + level // 2,
            "room_w":        6 + level // 3,
            "room_h":        5 + level // 3,
            "new_w":         1 if level < 5 else 2,
            "new_h":         1 if level < 7 else 2,
            "l_shape":       level >= 7,
            "walkway":       level >= 4,
            "q_type_pool":   self._q_type_pool(level),
        }

    @staticmethod
    def _q_type_pool(level):
        if level <= 1:
            return ["fit_basic"]
        if level <= 3:
            return ["fit_basic", "fit_against_wall"]
        if level <= 6:
            return ["fit_basic", "fit_against_wall",
                    "fit_adjacent", "walkway_clear"]
        # L7-9
        return ["fit_against_wall", "fit_adjacent",
                "walkway_clear", "pick_invalid",
                "fit_multi_constraint"]

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _overlaps(r1, r2):
        return not (r1[0] + r1[2] <= r2[0] or r2[0] + r2[2] <= r1[0] or
                    r1[1] + r1[3] <= r2[1] or r2[1] + r2[3] <= r1[1])

    def _overlap_any(self, piece, furniture):
        return any(self._overlaps(piece, f[1]) for f in furniture)

    @staticmethod
    def _in_room(piece, rw, rh):
        return (piece[0] >= 0 and piece[1] >= 0 and
                piece[0] + piece[2] <= rw and piece[1] + piece[3] <= rh)

    @staticmethod
    def _against_wall(piece, rw, rh):
        x, y, w, h = piece
        return x == 0 or y == 0 or (x + w) == rw or (y + h) == rh

    @staticmethod
    def _adjacent(p1, p2):
        """Share a boundary edge (touch but don't overlap)."""
        x1, y1, w1, h1 = p1
        x2, y2, w2, h2 = p2
        # Overlap range strictly positive on one axis and edge match on the other.
        touch_vertical = (x1 + w1 == x2 or x2 + w2 == x1) and (
            y1 < y2 + h2 and y2 < y1 + h1)
        touch_horizontal = (y1 + h1 == y2 or y2 + h2 == y1) and (
            x1 < x2 + w2 and x2 < x1 + w1)
        return touch_vertical or touch_horizontal

    # ------------------------------------------------------------------ #

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1028)
        style = self._random_style()

        for _ in range(30):
            result = self._try_make(rng, cfg, style, level)
            if result is not None:
                return result
        return None

    def _try_make(self, rng, cfg, style, level):
        rw, rh = cfg["room_w"], cfg["room_h"]
        nw, nh = cfg["new_w"], cfg["new_h"]

        # Place existing furniture
        furniture = []  # (name, (x,y,w,h))
        names_local = list(_FURNITURE_POOL)
        rng.shuffle(names_local)
        for i in range(cfg["n_pieces"]):
            for _ in range(80):
                fw = rng.randint(1, 2)
                fh = rng.randint(1, 2)
                fx = rng.randint(0, rw - fw)
                fy = rng.randint(0, rh - fh)
                piece = (fx, fy, fw, fh)
                if not self._overlap_any(piece, furniture):
                    furniture.append((names_local[i % len(names_local)], piece))
                    break

        # Optional walkway (rectangular strip that must remain clear of the new item)
        walkway = None
        if cfg["walkway"]:
            if rng.random() < 0.5:
                walkway = (0, rh // 2, rw, 1)  # horizontal strip
            else:
                walkway = (rw // 2, 0, 1, rh)  # vertical strip

        # Pick the question type
        q_type = rng.choice(cfg["q_type_pool"])
        constraint_target = None  # piece name to be adjacent to
        if q_type == "fit_adjacent" and furniture:
            constraint_target = rng.choice(furniture)[0]

        # Candidate positions — generate a bunch then filter 4 with useful mix.
        cand_positions = []
        tries = 0
        while len(cand_positions) < 30 and tries < 150:
            tries += 1
            cx_ = rng.randint(0, rw - nw)
            cy_ = rng.randint(0, rh - nh)
            p = (cx_, cy_, nw, nh)
            cand_positions.append(p)

        # Compute validity under q_type for each candidate.
        def check(p):
            if not self._in_room(p, rw, rh):
                return False
            if self._overlap_any(p, furniture):
                return False
            if cfg["walkway"] and walkway is not None and \
                    self._overlaps(p, walkway):
                return False
            if q_type == "fit_against_wall":
                return self._against_wall(p, rw, rh)
            if q_type == "fit_adjacent":
                target = next((f for f in furniture
                               if f[0] == constraint_target), None)
                return bool(target and self._adjacent(p, target[1]))
            if q_type == "walkway_clear":
                # already enforced
                return True
            if q_type == "pick_invalid":
                # valid means the base-constraint (not overlap + in room)
                # but question asks which IS invalid, so we invert.
                return True
            if q_type == "fit_multi_constraint":
                if not self._against_wall(p, rw, rh):
                    return False
                if furniture:
                    target = constraint_target or furniture[0][0]
                    t = next(f for f in furniture if f[0] == target)
                    return self._adjacent(p, t[1])
                return True
            # fit_basic
            return True

        # We need exactly 4 candidates with at least one valid (or invalid, for pick_invalid).
        # Dedup positions.
        seen = set()
        valid_list = []
        invalid_list = []
        for p in cand_positions:
            key = p
            if key in seen:
                continue
            seen.add(key)
            if check(p):
                valid_list.append(p)
            else:
                invalid_list.append(p)

        # For pick_invalid: need at least one invalid (must overlap something).
        # We check invalid under the strict no-overlap rule.
        def strict_ok(p):
            return self._in_room(p, rw, rh) and not self._overlap_any(p, furniture) \
                and (not cfg["walkway"] or walkway is None or not self._overlaps(p, walkway))

        if q_type == "pick_invalid":
            pv = [p for p in cand_positions if strict_ok(p)]
            pn = [p for p in cand_positions if not strict_ok(p)]
            # Re-dedup
            pv = list(dict.fromkeys(pv))
            pn = list(dict.fromkeys(pn))
            if not pn or len(pv) < 3:
                return None
            rng.shuffle(pv)
            rng.shuffle(pn)
            answer_piece = pn[0]
            distractor_pieces = pv[:3]
            all_pieces = [answer_piece] + distractor_pieces
            rng.shuffle(all_pieces)
            correct_idx = all_pieces.index(answer_piece)
        else:
            # Require EXACTLY one valid placement among the 4 candidates so
            # the problem is non-ambiguous. We must have >=1 valid and >=3
            # invalid (under the task-specific `check` predicate).
            if not valid_list or len(invalid_list) < 3:
                return None
            rng.shuffle(valid_list)
            rng.shuffle(invalid_list)
            answer_piece = valid_list[0]
            distractors = invalid_list[:3]
            all_pieces = [answer_piece] + distractors
            rng.shuffle(all_pieces)
            correct_idx = all_pieces.index(answer_piece)

        # If L-shape required, pad the piece footprint display but still use the
        # bounding box rectangle as the primary validity check.
        correct = "ABCD"[correct_idx]

        # Question text and image
        question = self._format_question(q_type, nw, nh, constraint_target)
        image = self._render(style, rng, rw, rh, furniture, walkway,
                             all_pieces, q_type, nw, nh, constraint_target,
                             cfg, level)
        return question, correct, image

    def _format_question(self, q_type, nw, nh, constraint_target):
        if q_type == "fit_basic":
            templates = [
                (f"A new {nw}x{nh} piece must be placed. Candidate positions "
                 f"A/B/C/D are shown as dashed outlines. Which fits without "
                 f"overlapping existing furniture?\n(A) (B) (C) (D)"),
                (f"Four possible placements for a {nw}x{nh} item are marked. "
                 f"Which placement avoids every existing piece?\n(A) (B) (C) (D)"),
            ]
        elif q_type == "fit_against_wall":
            templates = [
                (f"Four placements (A/B/C/D) for a new {nw}x{nh} piece are "
                 f"shown. Which placement sits against a wall AND does not "
                 f"overlap any existing furniture?\n(A) (B) (C) (D)"),
                (f"Choose the placement that is wall-adjacent and collision-"
                 f"free for a {nw}x{nh} item.\n(A) (B) (C) (D)"),
            ]
        elif q_type == "fit_adjacent":
            t = constraint_target or "an existing piece"
            templates = [
                (f"Which labeled placement puts the new {nw}x{nh} piece "
                 f"directly next to the {t} (sharing an edge, no overlap)?\n"
                 f"(A) (B) (C) (D)"),
                (f"Pick the position adjacent to the {t} while still fitting "
                 f"the room.\n(A) (B) (C) (D)"),
            ]
        elif q_type == "walkway_clear":
            templates = [
                ("The shaded strip is a walkway that must stay clear. "
                 "Which of the labeled placements does NOT block the walkway "
                 "and does not overlap any furniture?\n(A) (B) (C) (D)"),
                ("One placement keeps both the walkway and the furniture free."
                 " Which one?\n(A) (B) (C) (D)"),
            ]
        elif q_type == "pick_invalid":
            templates = [
                ("Three placements are legal; exactly ONE overlaps furniture "
                 "or exits the room. Which placement is INVALID?\n(A) (B) (C) (D)"),
                ("Find the placement that violates the constraints.\n(A) (B) (C) (D)"),
            ]
        else:  # fit_multi_constraint
            t = constraint_target or "an existing piece"
            templates = [
                (f"Which placement is both against a wall AND adjacent to "
                 f"the {t} (with no overlap)?\n(A) (B) (C) (D)"),
                (f"Select the placement that satisfies every listed "
                 f"constraint.\n(A) (B) (C) (D)"),
            ]
        r = self._rng
        return r.choice(templates) if r else templates[0]

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #

    def _render(self, style, rng, rw, rh, furniture, walkway,
                candidates, q_type, nw, nh, constraint_target,
                cfg, level):
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(max(7, rw + 2) * sc,
                                        max(6, rh + 2) * sc))
        fig.patch.set_facecolor(style["bg_color"])
        floor_palette = ["#fafaf0", "#f5f5dc", "#efe8d8", "#f0ead6",
                         "#f8f4e9", "#ece9d8"]
        ax.set_facecolor(rng.choice(floor_palette))
        ax.set_xlim(-1.0, rw + 1.0)
        ax.set_ylim(-1.0, rh + 1.0)
        ax.set_aspect("equal")

        # Room border
        wall_color = rng.choice(["#333", "#222", "#1a1a1a", "#2c3e50"])
        ax.add_patch(mpatches.Rectangle((0, 0), rw, rh, fill=False,
                                        edgecolor=wall_color, linewidth=3.5))

        # Optional walkway
        if walkway is not None:
            ax.add_patch(mpatches.Rectangle((walkway[0], walkway[1]),
                                            walkway[2], walkway[3],
                                            facecolor="#d0e4ff",
                                            edgecolor="#4c7ebf",
                                            linewidth=1.2, alpha=0.55,
                                            hatch=rng.choice(["//", "\\\\", "xx"])))
            ax.text(walkway[0] + walkway[2] / 2,
                    walkway[1] + walkway[3] / 2,
                    "Walkway", ha="center", va="center", fontsize=9,
                    color="#2c3e70", fontweight="bold", alpha=0.85)

        # Furniture
        palette = list(style["palette"])
        rng.shuffle(palette)
        for i, (name, (fx, fy, fw, fh)) in enumerate(furniture):
            color = palette[i % len(palette)]
            ax.add_patch(mpatches.Rectangle((fx, fy), fw, fh,
                                            facecolor=color,
                                            edgecolor=wall_color,
                                            linewidth=1.5, alpha=0.7))
            ax.text(fx + fw / 2, fy + fh / 2, name,
                    ha="center", va="center",
                    fontsize=max(10, style["font_size_base"] - 2),
                    fontweight="bold",
                    color="#1a1a1a")

        # Candidates with letter labels
        cand_colors = ["#e63946", "#1d3557", "#2a9d8f", "#e76f51"]
        rng.shuffle(cand_colors)
        for i, (px, py, pw, ph) in enumerate(candidates):
            c = cand_colors[i]
            ax.add_patch(mpatches.Rectangle(
                (px + 0.02, py + 0.02), pw - 0.04, ph - 0.04,
                fill=False, edgecolor=c, linewidth=2.5,
                linestyle="--"))
            # Position label in corner
            ax.text(px + 0.18, py + 0.18, chr(65 + i),
                    ha="center", va="center",
                    fontsize=max(13, style["font_size_base"] + 2),
                    fontweight="bold", color=c,
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white",
                              edgecolor=c, linewidth=1.6))

        # Title
        ax.axis("off")
        type_text = {
            "fit_basic": "Find a fit",
            "fit_against_wall": "Fit against a wall",
            "fit_adjacent": f"Fit adjacent to {constraint_target}",
            "walkway_clear": "Keep walkway clear",
            "pick_invalid": "Which is invalid?",
            "fit_multi_constraint": "Multi-constraint fit",
        }.get(q_type, "Room Layout")
        ax.set_title(f"Room {rw}x{rh} — {type_text}",
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold")
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = RoomLayoutReasoningQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer if ok else 'X'}")
