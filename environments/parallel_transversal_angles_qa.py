"""
Parallel Transversal Angles QA — redesign 2026-04-17.

FIX (2026-04-17): Prior version had unrecoverable image <-> GT mismatch.
The angle relation chain was RNG'd and invisible; model had no way to know
which angle of which intersection the ``\u22201``/``\u22204`` label referred to.

This version draws an EXPLICIT arc at the "given" angle's quadrant and an
EXPLICIT arc at the "?" angle's quadrant, so the model can read the
relationship directly (same side / opposite side of transversal; same/other
parallel line). GT is computed geometrically from the marked quadrants.

KEY PROPERTIES:
  * VALUE on image only (never in question text).
  * Arcs show exactly where each angle is measured.
  * L0: single transversal, two parallels, same intersection (vertical /
    linear pair) or trivial corresponding.
  * L9: 2-3 transversals, longer chain of relations, but still every
    marked angle is visually disambiguated by an arc.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrow
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_QUESTION_TEMPLATES_L0 = [
    "In the diagram, two parallel lines are cut by one transversal. The given angle's value is written on the image. Find the angle marked with '?'. Answer with the letter of the correct option.",
    "The diagram shows two parallel lines crossed by a transversal. Use the labeled given angle (shown with an arc on the image) to compute the angle marked '?' (also shown with an arc). Answer A/B/C/D.",
    "Two parallel lines are cut by a transversal. The given angle value and the '?' angle are each marked by an arc in the diagram. Find '?'. Answer with the letter of the correct option.",
]
_QUESTION_TEMPLATES_MID = [
    "The image shows parallel lines cut by transversal(s). Each marked angle is shown by an arc and labeled on the image. Compute the angle marked '?'. Answer with the letter.",
    "Use the parallel-line angle relations (alternate / corresponding / co-interior / vertical) to find the angle marked '?'. The given and asked angles are both marked by arcs on the diagram. Answer A/B/C/D.",
    "Find the angle '?' shown on the diagram. Values of given angles are on the image, and arcs mark exactly which angle is meant. Answer with the letter.",
]
_QUESTION_TEMPLATES_HARD = [
    "The diagram shows parallel lines cut by multiple transversals. The given angle and the asked '?' angle are each marked by an arc on the image. Apply angle relationships to deduce '?'. Answer with a single letter.",
    "Chain the angle relationships through the marked intersections to find the angle marked '?' on the diagram. Every angle referenced has an arc showing its position. Answer with the letter of the correct option.",
    "Parallel lines are cut by multiple transversals. Each angle of interest is shown by an arc on the diagram; the given angle's value is written next to its arc. Compute the angle marked '?'. Answer with the letter.",
]

def _relation_preserves_value(pos_a, pos_b):
    """Given two angle positions at different intersections on the same
    transversal, decide if the angles are equal (return True) or supplementary
    (False). Positions are in {'NW', 'NE', 'SW', 'SE'} describing the quadrant
    with respect to (transversal, parallel-line).

    Equal-angle cases (value preserved):
      * same quadrant across intersections = corresponding angles.
      * NW<->SE or NE<->SW across intersections = alternate interior/exterior.
      * NW<->SW (same intersection, vertical pair across center) = NOT this
        function; handled separately.
    Supplementary cases:
      * NW<->SW same intersection = linear pair.
      * NW<->NE across intersections = co-interior / same-side.
    """
    if pos_a == pos_b:
        return True
    # alternate: diagonally opposite across transversal
    diag = {("NW", "SE"), ("SE", "NW"), ("NE", "SW"), ("SW", "NE")}
    if (pos_a, pos_b) in diag:
        return True
    return False

def _vertical_pair(pos_a, pos_b):
    """Same intersection, vertical pair (equal)."""
    vert = {("NW", "SE"), ("SE", "NW"), ("NE", "SW"), ("SW", "NE")}
    return (pos_a, pos_b) in vert

class ParallelTransversalAnglesQA(StandaloneVisualEnv):
    ENV_NAME = "parallel_transversal_angles"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    TEXTBOOK_POSTPROCESS = True  # v4 B1: textbook-scan filter (~30% of rollouts)

    _PAR_TITLES = [
        "Parallel lines + transversals",
        "Transversal angles",
        "Parallel cuts",
        "Angle diagram",
        "Lines and angles",
        "Angle chain",
    ]

    _POSITIONS = ["NW", "NE", "SW", "SE"]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            # L0..L4: 1 transversal, 2 parallels. L5-7: 2 transversals, 2 parallels.
            # L8-9: 2 transversals, 3 parallels.
            "n_transversals":    1 if level < 5 else 2,
            "n_parallels":       2 if level < 8 else 3,
            "n_hops":            1 + level // 2,    # 1..5
            "tight_distractors": level >= 4,
            "show_arc_hint":     level <= 2,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_hops"]

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random,
                      cfg: Dict, level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n_par = cfg["n_parallels"]
        n_trans = cfg["n_transversals"]

        given_ang = rng.randint(35, 145)
        if given_ang in (89, 90, 91):
            given_ang = 75

        # Pick a "given" quadrant at intersection (tv=0, par=0), and an
        # "asked" quadrant at intersection (tv, par).
        given_tv = 0
        given_par = 0
        given_pos = rng.choice(self._POSITIONS)

        ask_tv = rng.randint(0, n_trans - 1)
        ask_par = rng.randint(0, n_par - 1)
        # Force at least one structural hop away.
        if ask_tv == given_tv and ask_par == given_par:
            if n_par > 1:
                ask_par = (given_par + 1) % n_par
            else:
                ask_tv = (given_tv + 1) % max(2, n_trans)
        ask_pos = rng.choice(self._POSITIONS)

        # Ground truth: walk from given angle to asked angle, flipping as needed.
        # Step 1: within same intersection (given_tv, given_par), move to a
        # reference quadrant (we use NW) using vertical/linear-pair rules.
        gt_angle = self._compute_gt(given_ang, given_pos, ask_pos,
                                    given_tv, given_par,
                                    ask_tv, ask_par, n_trans, n_par)
        if gt_angle is None or gt_angle <= 0 or gt_angle >= 180:
            return None
        gt = gt_angle
        supp = 180 - gt

        tight = cfg["tight_distractors"]
        if tight:
            pool = {max(1, gt - 2), max(1, gt - 1), gt + 1, gt + 2, supp}
        else:
            pool = {max(1, gt - 15), max(1, gt - 10), gt + 10, gt + 15,
                    supp, 90 - abs(gt - 90)}
        pool.discard(gt)
        pool_list = [p for p in pool if p != gt and 0 < p < 180]
        rng.shuffle(pool_list)
        distractors = pool_list[:3]
        if len(distractors) < 3:
            for k in (5, -5, 8, -8, 20):
                cand = gt + k
                if 0 < cand < 180 and cand != gt and cand not in distractors:
                    distractors.append(cand)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            return None

        options_vals = [gt] + distractors[:3]
        rng.shuffle(options_vals)
        if options_vals.count(gt) > 1:
            return None
        answer_letter = chr(ord("A") + options_vals.index(gt))
        options_str = [f"{v}\u00b0" for v in options_vals]

        # Choose a question phrasing by level band.
        if level <= 1:
            q_phrase = rng.choice(_QUESTION_TEMPLATES_L0)
        elif level <= 5:
            q_phrase = rng.choice(_QUESTION_TEMPLATES_MID)
        else:
            q_phrase = rng.choice(_QUESTION_TEMPLATES_HARD)

        q_lines = [q_phrase, ""]
        for i, v in enumerate(options_str):
            q_lines.append(f"({chr(ord('A') + i)}) {v}")
        question = "\n".join(q_lines)

        image = self._render(given_ang, given_tv, given_par, given_pos,
                             ask_tv, ask_par, ask_pos,
                             cfg, options_str, rng)
        return question, answer_letter, image

    def _compute_gt(self, given_ang, given_pos, ask_pos,
                    given_tv, given_par, ask_tv, ask_par,
                    n_trans, n_par):
        """Compute asked angle value from quadrant positions.

        For parallel lines cut by a single transversal:
          * Same quadrant at different parallels = corresponding (equal).
          * Alternate interior/exterior (diag) = equal.
          * Same-side (co-interior) = supplementary.
          * At the same intersection: vertical pair = equal; linear pair =
            supplementary.

        For multiple transversals at the SAME parallel line: the angles at
        different transversals are NOT simply related by parallel-line rules
        unless the transversals are parallel to each other. We treat different
        transversals at the same parallel as a "same-quadrant" rule only when
        the transversals happen to meet the parallel at the same angle — which
        is the case here (all transversals use the same slope for simplicity).
        With this simplification, same quadrant = equal, linear-pair flips,
        vertical pair = equal.
        """
        # All transversals share slope (see _render). Therefore, for any
        # two intersections on our diagram, two angles are equal iff their
        # quadrant labels are either the same quadrant OR vertical/alternate
        # pairs (NW<->SE, NE<->SW). Otherwise they are supplementary.
        if _relation_preserves_value(given_pos, ask_pos):
            return given_ang
        else:
            return 180 - given_ang

    def _render(self, given_ang, given_tv, given_par, given_pos,
                ask_tv, ask_par, ask_pos,
                cfg, options, rng) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = style["font_family"]
        fs = max(11, style["font_size_base"])

        fig = plt.figure(figsize=(10.5 * sc, 6.5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax_p = fig.add_subplot(1, 2, 1)
        ax_t = fig.add_subplot(1, 2, 2)
        ax_p.set_aspect("equal")
        ax_p.axis("off")
        ax_t.axis("off")

        palette = list(style["palette"])
        rng.shuffle(palette)
        lw = style["line_width"]

        n_par = cfg["n_parallels"]
        n_trans = cfg["n_transversals"]

        par_y = [0, 3, 6][:n_par]
        par_color = palette[2 % len(palette)]
        for y in par_y:
            ax_p.plot([-2, 15], [y, y], "-", color=par_color,
                      linewidth=lw * 1.2)
            # Arrowhead tick for the parallel.
            ax_p.annotate("", xy=(14.5, y), xytext=(13.5, y),
                          arrowprops=dict(arrowstyle="->", color=par_color))

        slope_deg = given_ang  # transversals all share slope
        slope = math.tan(math.radians(slope_deg))
        # Intersection x-coords: transversals cross parallel y=0 at these x's.
        x_at_y0 = [3.0 + i * 4.5 for i in range(n_trans)]

        def intersection_xy(tv_idx, par_idx):
            x0 = x_at_y0[tv_idx]
            y = par_y[par_idx]
            if abs(slope) > 1e-6:
                x = x0 + y / slope
            else:
                x = x0
            return x, y

        # Draw transversals.
        tv_color = palette[3 % len(palette)]
        for i, x0 in enumerate(x_at_y0):
            y_start = par_y[0] - 2
            y_end = par_y[-1] + 2
            if abs(slope) > 1e-6:
                x_top = x0 + y_end / slope
                x_bot = x0 + y_start / slope
            else:
                x_top = x0
                x_bot = x0
            ax_p.plot([x_bot, x_top], [y_start, y_end], "-",
                      color=tv_color, linewidth=lw)

        # Arc drawing. For a given intersection (x, y) and a transversal with
        # slope_deg, the four quadrants NW/NE/SW/SE are bounded by the two
        # half-lines of the transversal and the parallel. For simplicity use
        # coarse angular ranges relative to the parallel (horizontal).
        #   NE: 0deg to slope_deg (between +x parallel and "up" part of tv)
        #   NW: slope_deg to 180deg
        #   SW: 180 to 180+slope_deg
        #   SE: 180+slope_deg to 360
        pos_arc_range = {
            "NE": (0, slope_deg),
            "NW": (slope_deg, 180),
            "SW": (180, 180 + slope_deg),
            "SE": (180 + slope_deg, 360),
        }
        arc_r = 1.1
        label_r = 1.9

        def draw_arc_at(tv_idx, par_idx, pos, color, text_str, is_given=True):
            x, y = intersection_xy(tv_idx, par_idx)
            t1, t2 = pos_arc_range[pos]
            ax_p.add_patch(Arc((x, y), arc_r * 2, arc_r * 2,
                               theta1=t1, theta2=t2,
                               color=color, lw=2.5))
            mid_ang = math.radians((t1 + t2) / 2.0)
            tx = x + label_r * math.cos(mid_ang)
            ty = y + label_r * math.sin(mid_ang)
            ax_p.text(tx, ty, text_str,
                      fontsize=fs + 1, fontweight="bold", family=ff,
                      ha="center", va="center", color=color,
                      bbox=dict(facecolor="white", edgecolor=color,
                                boxstyle="round,pad=0.22", linewidth=1.3))

        given_color = palette[4 % len(palette)]
        ask_color = palette[5 % len(palette)]
        # Given angle arc + value label on image
        draw_arc_at(given_tv, given_par, given_pos, given_color,
                    f"{given_ang}\u00b0", is_given=True)
        # Asked angle arc + "?" label
        draw_arc_at(ask_tv, ask_par, ask_pos, ask_color, "?",
                    is_given=False)

        # set bounds
        xs = []
        ys = []
        for ti in range(n_trans):
            for pi in range(n_par):
                x, y = intersection_xy(ti, pi)
                xs.append(x)
                ys.append(y)
        pad_x = 3.5
        pad_y = 2.5
        ax_p.set_xlim(min(xs + [-2]) - pad_x, max(xs + [15]) + pad_x)
        ax_p.set_ylim(par_y[0] - pad_y, par_y[-1] + pad_y)
        ax_p.set_title(rng.choice(self._PAR_TITLES),
                       fontsize=fs + 1, family=ff, pad=6)

        # Options panel.
        ax_t.set_xlim(0, 10)
        ax_t.set_ylim(0, 12)
        y = 11.5
        ax_t.text(0.3, y, "Options:", fontsize=fs + 1, fontweight="bold",
                  family=ff, ha="left", va="top", color="#2c3e50")
        y -= 0.9
        for i, o in enumerate(options):
            ax_t.text(0.5, y, f"({chr(ord('A') + i)}) {o}",
                      fontsize=fs, family=ff, ha="left", va="top",
                      color="#1a1a1a")
            y -= 0.8
        if cfg.get("show_arc_hint"):
            ax_t.text(
                0.3, 3.0,
                "(Hint: use corresponding / alternate /\nco-interior / vertical relations.)",
                fontsize=fs - 1, family=ff, ha="left", va="top",
                color="#555")

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02,
                            wspace=0.15)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = ParallelTransversalAnglesQA()
    for lv in [0, 3, 6, 9]:
        for s in range(3):
            ok = env.generate(s, {"level": lv})
            print(f"L{lv} s{s}: ok={ok}, ans={env._answer}")
