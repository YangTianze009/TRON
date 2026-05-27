"""
Figure to Quantity Table QA (v4 G1b).

Complement to geometry_label_reading: here the task is to output a full
dict of all labeled quantities in the figure, not just one. This forces
parsing EVERY label.

Task: render a figure (cylinder, triangle, rect prism) with 2-4 labeled
quantities; output the full {role: value} dict.

Reward: structured dict equality with tolerant numeric compare.

Level axes:
  A) Number of quantities: 2 at L0, 3 at L3, 4 at L6+
"""
import random
import math
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "Parse every labeled quantity in the figure. Output a dict mapping each role to its labeled value: {{'role': value, ...}}. Use the exact role names visible (e.g., 'radius', 'height', 'diameter', 'length', 'width', 'side_AB'). Put the dict in <answer>...</answer>.",
    "Read all labels from the figure and produce a dict {{role: value}}. Put the complete dict in <answer>...</answer>.",
    "Output the complete labeled-quantities dict from the figure. Put in <answer>...</answer>.",
    "Produce a dict mapping every visible label to its role. Put in <answer>...</answer>.",
    "Parse all figure labels and output as a dict {{role: value}}. Put in <answer>...</answer>.",
    "Give the full dict of (role, value) pairs from the figure's labels. Put in <answer>...</answer>.",
    "Extract every label and format as {{role: value}} dict. Put in <answer>...</answer>.",
    "Read all labeled quantities; output {{role: value}} dict. Put in <answer>...</answer>.",
    "Full labeled-quantities dict? Put in <answer>...</answer>.",
    "Dict of {{role: value}} from the figure labels? Put in <answer>...</answer>.",
    "Output {{role: value}} for every labeled quantity. Put in <answer>...</answer>.",
    "Parse the figure and give the full {{role: value}} mapping. Put in <answer>...</answer>.",
    "{{role: value}} dict for all labels in the figure. Put in <answer>...</answer>.",
    "Full label parse: {{role: value}} dict. Put in <answer>...</answer>.",
    "Output the dict of all labeled quantities. Put in <answer>...</answer>.",
    "Read every label; produce {{role: value}} dict. Put in <answer>...</answer>.",
]

class FigureToQuantityTableQA(StandaloneVisualEnv):
    ENV_NAME = "figure_to_quantity_table"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_quant = min(5, 2 + level // 3)
        # show_role_text: at low levels we annotate each value with its role
        # name (e.g. "radius = 4") so the model just needs OCR; at higher
        # levels only the bare value is shown and the model must infer the
        # role from its position in the figure.
        if level <= 2:
            show_role_text = True   # always show role text
        elif level <= 5:
            show_role_text = "partial"  # show role text for some labels
        else:
            show_role_text = False  # bare values, infer roles from position
        return {"n_quant": n_quant, "show_role_text": show_role_text,
                "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 509)
        self._primary_complexity_feature = level

        n = cfg["n_quant"]
        # Pick a figure type based on n
        if n == 2:
            fig_type = rng.choice(["rectangle", "circle", "triangle"])
        elif n == 3:
            fig_type = rng.choice(["triangle", "cylinder", "trapezoid"])
        else:
            fig_type = rng.choice(["cylinder_net", "cone", "rectangular_prism"])

        if fig_type == "rectangle":
            w, h = rng.randint(3, 12), rng.randint(2, 8)
            roles = {"width": w, "height": h}
        elif fig_type == "circle":
            r = rng.randint(2, 10)
            roles = {"radius": r, "diameter": r * 2}
        elif fig_type == "triangle":
            ab, ac, bc = rng.randint(3, 10), rng.randint(3, 10), rng.randint(3, 10)
            roles = {"side_AB": ab, "side_AC": ac}
            if n >= 3:
                roles["side_BC"] = bc
        elif fig_type == "cylinder":
            r, h = rng.randint(2, 6), rng.randint(3, 10)
            roles = {"radius": r, "height": h, "diameter": r * 2}
        elif fig_type == "trapezoid":
            top, bot, h = rng.randint(3, 8), rng.randint(8, 14), rng.randint(2, 6)
            roles = {"top_base": top, "bottom_base": bot, "height": h}
        elif fig_type == "cylinder_net":
            r, h = rng.randint(2, 6), rng.randint(3, 10)
            roles = {"radius": r, "height": h, "diameter": r * 2,
                     "circumference": round(2 * math.pi * r, 2)}
        elif fig_type == "cone":
            r, h, slant = rng.randint(2, 5), rng.randint(3, 7), 0
            slant = round(math.sqrt(r * r + h * h), 2)
            roles = {"radius": r, "height": h, "slant_height": slant,
                     "diameter": r * 2}
        else:  # rectangular_prism
            l, w, h = rng.randint(3, 10), rng.randint(3, 10), rng.randint(2, 8)
            roles = {"length": l, "width": w, "height": h}
            if n >= 4:
                roles["volume"] = l * w * h

        # Trim to n
        keys = list(roles.keys())[:n]
        roles = {k: roles[k] for k in keys}

        answer = str(roles)
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx]

        img = self._render(fig_type, roles, rng, cfg["show_role_text"])
        return q, answer, img

    def _render(self, fig_type, roles, rng, show_role_text=False):
        # Labels can be either a bare value ("4") forcing the model to infer
        # the role from position, or "role = value" annotations that turn
        # the task into pure OCR. Difficulty axis controlled by show_role_text.
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 8); ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        # Determine which labels include the role name.
        role_names = list(roles.keys())
        if show_role_text is True:
            label_with_role = set(role_names)
        elif show_role_text == "partial":
            # Half the labels carry the role name (round up).
            n_with_role = (len(role_names) + 1) // 2
            shuffled = list(role_names)
            rng.shuffle(shuffled)
            label_with_role = set(shuffled[:n_with_role])
        else:
            label_with_role = set()

        def _fmt(role, v):
            if role in label_with_role:
                return f"{role} = {v}"
            return f"{v}"

        def _lbl_role(x, y, role, v, ha="center", va="center"):
            ax.text(x, y, _fmt(role, v), fontsize=12, ha=ha, va=va,
                    fontweight="bold",
                    bbox=dict(facecolor="lightyellow",
                              edgecolor="gray", pad=2))

        # Backwards-compatible plain value label (used by code paths below
        # that don't carry a role name; kept for safety).
        def _lbl(x, y, v, ha="center", va="center"):
            ax.text(x, y, f"{v}", fontsize=12, ha=ha, va=va,
                    fontweight="bold",
                    bbox=dict(facecolor="lightyellow", edgecolor="gray", pad=2))

        if fig_type == "rectangle":
            # width = 6, height = 3 box at (1,2)-(7,5)
            ax.add_patch(mpatches.Rectangle((1, 2), 6, 3, fc="none",
                                             ec="black", lw=2.0))
            if "width" in roles:
                _lbl_role(4, 1.4, "width", roles["width"])  # below bottom edge
            if "height" in roles:
                _lbl_role(7.6, 3.5, "height", roles["height"], ha="left")  # right of right edge
        elif fig_type == "circle":
            ax.add_patch(mpatches.Circle((4, 4), 2.2, fc="none",
                                          ec="black", lw=2.0))
            if "radius" in roles:
                # radius line from center to right edge
                ax.plot([4, 6.2], [4, 4], color="black", lw=1.0,
                        linestyle="--")
                _lbl_role(5.1, 4.4, "radius", roles["radius"])  # midpoint of radius line
            if "diameter" in roles:
                # diameter line across bottom half
                ax.plot([1.8, 6.2], [3.1, 3.1], color="gray", lw=1.0,
                        linestyle="--")
                _lbl_role(4, 2.6, "diameter", roles["diameter"])  # below diameter chord
        elif fig_type == "triangle":
            verts = [(1, 1), (7, 1), (4, 6)]
            ax.add_patch(mpatches.Polygon(verts, fc="none", ec="black", lw=2.0))
            ax.text(0.6, 0.6, "A", fontsize=11, fontweight="bold")
            ax.text(7.2, 0.6, "B", fontsize=11, fontweight="bold")
            ax.text(4, 6.3, "C", fontsize=11, fontweight="bold")
            if "side_AB" in roles:
                _lbl_role(4, 0.5, "side_AB", roles["side_AB"])  # along bottom AB
            if "side_AC" in roles:
                _lbl_role(2.0, 3.7, "side_AC", roles["side_AC"], ha="right")  # left side AC
            if "side_BC" in roles:
                _lbl_role(6.0, 3.7, "side_BC", roles["side_BC"], ha="left")  # right side BC
        elif fig_type == "cylinder":
            # drawn with sides x=2,6 and ellipses y=1,5
            ax.plot([2, 2], [1, 5], color="black", lw=2)
            ax.plot([6, 6], [1, 5], color="black", lw=2)
            ax.add_patch(mpatches.Ellipse((4, 5), 4, 0.8, fc="none",
                                           ec="black", lw=2))
            ax.add_patch(mpatches.Ellipse((4, 1), 4, 0.8, fc="none",
                                           ec="black", lw=2))
            if "radius" in roles:
                # radius line on top ellipse (center to right)
                ax.plot([4, 6], [5, 5], color="black", lw=1.0, linestyle="--")
                _lbl_role(5, 5.6, "radius", roles["radius"])  # above radius line
            if "diameter" in roles:
                # diameter along bottom ellipse
                ax.plot([2, 6], [0.3, 0.3], color="gray", lw=1.0, linestyle="--")
                _lbl_role(4, 0.0, "diameter", roles["diameter"])
            if "height" in roles:
                # vertical height label on right
                _lbl_role(6.8, 3, "height", roles["height"], ha="left")
        elif fig_type == "trapezoid":
            # (1,1)(7,1)(5.5,4)(2.5,4): top_base=3, bottom_base=6, height=3
            ax.add_patch(mpatches.Polygon([(1, 1), (7, 1), (5.5, 4), (2.5, 4)],
                                            fc="none", ec="black", lw=2.0))
            if "bottom_base" in roles:
                _lbl_role(4, 0.5, "bottom_base", roles["bottom_base"])
            if "top_base" in roles:
                _lbl_role(4, 4.4, "top_base", roles["top_base"])
            if "height" in roles:
                _lbl_role(7.5, 2.5, "height", roles["height"], ha="left")
        elif fig_type == "cylinder_net":
            # unrolled: rectangle (1,2)-(6,5) for side + circle at (6.8,3.5)
            ax.add_patch(mpatches.Rectangle((1, 2), 5, 3, fc="none",
                                             ec="black", lw=2))
            ax.add_patch(mpatches.Circle((6.8, 3.5), 0.8, fc="none",
                                           ec="black", lw=2))
            if "circumference" in roles:
                _lbl_role(3.5, 1.4, "circumference", roles["circumference"])  # below rectangle bottom
            if "height" in roles:
                _lbl_role(0.5, 3.5, "height", roles["height"], ha="right")  # left of rectangle
            if "radius" in roles:
                ax.plot([6.8, 7.6], [3.5, 3.5], color="black", lw=1.0,
                        linestyle="--")
                _lbl_role(7.2, 3.9, "radius", roles["radius"])  # on circle's radius line
            if "diameter" in roles:
                _lbl_role(6.8, 2.3, "diameter", roles["diameter"])  # below circle
        elif fig_type == "cone":
            # apex (4,6), base diameter 6 at y=1
            ax.plot([1, 4, 7], [1, 6, 1], color="black", lw=2)
            ax.add_patch(mpatches.Ellipse((4, 1), 6, 0.8, fc="none",
                                           ec="black", lw=2))
            if "radius" in roles:
                ax.plot([4, 7], [1, 1], color="black", lw=1.0, linestyle="--")
                _lbl_role(5.5, 0.5, "radius", roles["radius"])  # along radius
            if "diameter" in roles:
                _lbl_role(4, -0.2, "diameter", roles["diameter"])  # below base
            if "height" in roles:
                ax.plot([4, 4], [1, 6], color="gray", lw=1.0, linestyle=":")
                _lbl_role(4.3, 3.5, "height", roles["height"], ha="left")  # along axis
            if "slant_height" in roles:
                _lbl_role(5.8, 3.5, "slant_height", roles["slant_height"], ha="left")  # along right slant
        else:  # rectangular_prism
            ax.add_patch(mpatches.Rectangle((1, 1), 4, 3, fc="none",
                                             ec="black", lw=2))
            ax.plot([1, 2.5, 2.5, 1], [1, 2.5, 2.5, 1], color="black", lw=1)
            ax.plot([5, 6.5], [1, 2.5], color="black", lw=1)
            ax.plot([5, 6.5], [4, 5.5], color="black", lw=1)
            ax.plot([2.5, 6.5], [5.5, 5.5], color="black", lw=1)
            ax.add_patch(mpatches.Rectangle((2.5, 2.5), 4, 3, fc="none",
                                             ec="black", lw=1.5))
            if "length" in roles:
                _lbl_role(3, 0.5, "length", roles["length"])  # along bottom front edge
            if "width" in roles:
                _lbl_role(0.5, 2.5, "width", roles["width"], ha="right")  # left front edge
            if "height" in roles:
                _lbl_role(5.5, 1.6, "height", roles["height"], ha="left")  # depth edge
            if "volume" in roles:
                _lbl_role(4.5, 4, "volume", roles["volume"])  # inside prism

        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        import ast, json, re
        pred = predicted.strip().rstrip(".").rstrip(",")
        gt = ground_truth.strip().rstrip(".").rstrip(",")
        if pred.lower() == gt.lower():
            return True

        # Strip markdown code-block fences if present
        pred = re.sub(r"```[^\n]*\n|```", "", pred)

        def _parse_dict(s: str):
            # Try several parsers in order
            # 1. ast.literal_eval (Python dict literal)
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
            # 2. JSON
            try:
                return json.loads(s)
            except Exception:
                pass
            # 3. Replace single→double quotes then JSON
            s2 = s.replace("'", '"')
            try:
                return json.loads(s2)
            except Exception:
                pass
            return None

        try:
            gt_d = ast.literal_eval(gt)
        except Exception:
            try:
                gt_d = json.loads(gt.replace("'", '"'))
            except Exception:
                return False
        if not isinstance(gt_d, dict):
            return False

        # Locate a dict-like substring in predicted (allow multi-line, may
        # contain nested braces — take the longest balanced "{…}" block).
        candidates = []
        depth = 0; start = -1
        for i, ch in enumerate(pred):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    candidates.append(pred[start:i + 1])
                    start = -1
        # also include the entire predicted as a fallback
        candidates.append(pred)

        for cand in candidates:
            d = _parse_dict(cand)
            if not isinstance(d, dict):
                continue
            # Compare key-by-key (case-insensitive on string keys, numeric
            # tolerance on values).
            d_norm = {str(k).strip().lower(): v for k, v in d.items()}
            gt_norm = {str(k).strip().lower(): v for k, v in gt_d.items()}
            if set(d_norm.keys()) != set(gt_norm.keys()):
                continue
            ok = True
            for k, gv in gt_norm.items():
                pv = d_norm.get(k)
                if isinstance(gv, (int, float)) and isinstance(pv, (int, float)):
                    if abs(pv - gv) > 0.1 and abs(pv - gv) / max(abs(gv), 1e-9) > 0.02:
                        ok = False; break
                else:
                    try:
                        if abs(float(pv) - float(gv)) > 0.1:
                            ok = False; break
                    except Exception:
                        if str(pv).strip().lower() != str(gv).strip().lower():
                            ok = False; break
            if ok:
                return True
        return False

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_fqt"
    os.makedirs(out_dir, exist_ok=True)
    env = FigureToQuantityTableQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 281
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[fqt L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/fqt_s{s}_L{level}.png")
            print(f"[fqt L{level} s{s}] A={env._answer}")
