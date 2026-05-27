"""
Compound 3D Volume QA.

Targets: Solid Geometry, dynamic-math solid geometry.
Capabilities: M2 (solid geometry) + X1 (multi-step).

Visual: a compound solid (e.g. a cube with a smaller cube on top, a
cylinder with a hemisphere on top, a cylinder plus a cone plus a
hemisphere) rendered isometrically. Each component's dimensions are
labelled.

Task: MCQ / integer -- total volume of the compound shape.

Difficulty axes:
  1. n_components = 2 + level // 3 (2..5 components combined).
  2. operation_type: L0 = addition only (prisms stacked); L5 = addition +
     subtraction (e.g. cylinder with hemisphere scooped out); L9 =
     addition + subtraction + shared-dimension inference, with a mix of
     prism / cone / sphere / cylinder.

parameter = {"level": int in [0,9]}
"""
import math
import random
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TITLES = [
    "Compound solid",
    "Composite 3D figure",
    "Combined solid",
    "Compound shape",
]

class Compound3dVolumeQA(StandaloneVisualEnv):
    ENV_NAME = "compound_3d_volume"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # ------------------------------------------------------------------ #
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, int(level)))
        n_components = 2 + level // 3       # 2..5
        if level <= 3:
            operation_type = "addition"
        elif level <= 6:
            operation_type = "subtraction"
        else:
            operation_type = "subtraction_and_derived"
        if level <= 3:
            types = ["prism"]
        elif level <= 5:
            types = ["prism", "cylinder"]
        elif level <= 7:
            types = ["prism", "cylinder", "cone"]
        else:
            types = ["prism", "cylinder", "cone", "sphere"]
        return {
            "n_components": n_components,
            "operation_type": operation_type,
            "component_types": types,
            "answer_format": "pi" if level >= 4 else "integer",
        }

    # ------------------------------------------------------------------ #
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1031)

        # ------- Build component list -------
        # Each component is a tuple: (kind, params_dict, sign, volume_int_part, volume_pi_coef)
        # volume = volume_int_part + volume_pi_coef * pi
        components = []
        need_pi = cfg["answer_format"] == "pi"

        # Main body - always a prism with known dimensions
        a = sub_rng.randint(3, 5)
        b = sub_rng.randint(3, 5)
        c = sub_rng.randint(3, 6)
        base_vol_int = a * b * c
        base_vol_pi = 0.0
        components.append(("prism",
                           {"a": a, "b": b, "c": c, "x": 0, "y": 0, "z": 0},
                           +1, base_vol_int, base_vol_pi))
        prev_top_z = c

        for i in range(cfg["n_components"] - 1):
            allowed = list(cfg["component_types"])
            kind = sub_rng.choice(allowed)
            if kind == "prism":
                sa = sub_rng.randint(1, max(1, a - 1))
                sb = sub_rng.randint(1, max(1, b - 1))
                sc = sub_rng.randint(1, 3)
                v_int = sa * sb * sc
                v_pi = 0.0
                params = {"a": sa, "b": sb, "c": sc,
                          "x": (a - sa) / 2.0,
                          "y": (b - sb) / 2.0,
                          "z": prev_top_z}
                sign = +1
                prev_top_z += sc
            elif kind == "cylinder":
                rr = sub_rng.randint(1, max(1, min(a, b) // 2))
                hh = sub_rng.randint(2, 4)
                v_int = 0.0
                v_pi = rr * rr * hh
                params = {"r": rr, "h": hh,
                          "cx": a / 2.0, "cy": b / 2.0, "z": prev_top_z}
                sign = +1
                prev_top_z += hh
            elif kind == "cone":
                rr = sub_rng.randint(1, max(1, min(a, b) // 2))
                hh = sub_rng.randint(2, 4)
                v_int = 0.0
                v_pi = rr * rr * hh / 3.0
                params = {"r": rr, "h": hh,
                          "cx": a / 2.0, "cy": b / 2.0, "z": prev_top_z}
                sign = +1
                prev_top_z += hh
            else:  # sphere/hemisphere
                rr = sub_rng.randint(1, max(1, min(a, b) // 2))
                # treat as hemisphere sitting on top; V = (2/3)*pi*r^3
                v_int = 0.0
                v_pi = 2.0 * rr ** 3 / 3.0
                params = {"r": rr, "cx": a / 2.0,
                          "cy": b / 2.0, "z": prev_top_z}
                sign = +1
                prev_top_z += rr
            components.append((kind, params, sign, v_int, v_pi))

        # For subtraction levels: convert last component into a subtracted
        # cavity scooped from the main base.
        if cfg["operation_type"] in ("subtraction",
                                     "subtraction_and_derived"):
            if len(components) >= 2:
                last_kind, last_p, _sign, lv_int, lv_pi = components[-1]
                # Convert last component into a scooped-out cavity in the
                # base, keeping its params. Adjust z so it sits at the
                # bottom of the base.
                if last_kind == "prism":
                    last_p["z"] = 0.0
                elif last_kind in ("cylinder", "cone"):
                    last_p["z"] = 0.0
                elif last_kind == "sphere":
                    last_p["z"] = 0.0
                components[-1] = (last_kind, last_p, -1, lv_int, lv_pi)

        # Total volume
        total_int = 0.0
        total_pi = 0.0
        for kind, params, sign, vi, vp in components:
            total_int += sign * vi
            total_pi += sign * vp

        # Format answer
        total_float = total_int + total_pi * math.pi
        int_is_clean = abs(total_int - round(total_int)) < 1e-9
        pi_is_clean = abs(total_pi - round(total_pi)) < 1e-9
        neg_pi = total_pi < 0
        if need_pi:
            # Format like "45pi + 36" or "(134/3)pi"
            if abs(total_pi) < 1e-9:
                correct_str = f"{int(round(total_int))}" if int_is_clean else f"{round(total_int, 2)}"
            elif abs(total_int) < 1e-9:
                if pi_is_clean:
                    correct_str = f"{int(round(total_pi))}pi"
                else:
                    correct_str = f"{round(total_pi, 2)}pi"
            else:
                pi_part = (f"{int(round(total_pi))}pi"
                           if pi_is_clean else f"{round(total_pi, 2)}pi")
                int_part = (f"{int(round(total_int))}"
                            if int_is_clean else f"{round(total_int, 2)}")
                correct_str = f"{pi_part} + {int_part}"
        else:
            # all integer components (prisms only), just int
            correct_str = f"{int(round(total_int + total_pi * math.pi))}"

        # Distractors - common errors
        cand = []
        # Error: sum ignores sign
        abs_int = sum(vi for (_, _, _, vi, _) in components)
        abs_pi = sum(vp for (_, _, _, _, vp) in components)
        cand.append((abs_int, abs_pi))
        # Error: swap sign (subtract everything)
        cand.append((total_int * -1 if total_int != 0 else total_int + 4,
                     total_pi * -1 if total_pi != 0 else total_pi + 4))
        # Error: include/exclude cone factor (1/3)
        cone_vp = 0.0
        for kind, params, sign, vi, vp in components:
            if kind == "cone":
                cone_vp += sign * vp * 2   # replace vp with vp*3 by adding 2*vp
        cand.append((total_int, total_pi + cone_vp))
        # Error: forget the 2/3 on hemisphere
        sphere_vp = 0.0
        for kind, params, sign, vi, vp in components:
            if kind == "sphere":
                # if treated as full sphere (4/3 pi r^3) instead of 2/3
                sphere_vp += sign * vp     # doubling effectively
        cand.append((total_int, total_pi + sphere_vp))
        # Error: include only first component
        k0, p0, s0, vi0, vp0 = components[0]
        cand.append((vi0 * s0, vp0 * s0))
        # Error: double the first
        cand.append((total_int + vi0, total_pi + vp0))

        def _fmt_pair(pair):
            ti, tp = pair
            if need_pi:
                ti_clean = abs(ti - round(ti)) < 1e-9
                tp_clean = abs(tp - round(tp)) < 1e-9
                if abs(tp) < 1e-9:
                    return (f"{int(round(ti))}" if ti_clean
                            else f"{round(ti, 2)}")
                if abs(ti) < 1e-9:
                    return (f"{int(round(tp))}pi" if tp_clean
                            else f"{round(tp, 2)}pi")
                pi_part = (f"{int(round(tp))}pi" if tp_clean
                           else f"{round(tp, 2)}pi")
                int_part = (f"{int(round(ti))}" if ti_clean
                            else f"{round(ti, 2)}")
                return f"{pi_part} + {int_part}"
            else:
                return f"{int(round(ti + tp * math.pi))}"

        distractors = []
        seen = {correct_str}
        for p in cand:
            s = _fmt_pair(p)
            if s not in seen:
                distractors.append(s)
                seen.add(s)
            if len(distractors) >= 3:
                break
        while len(distractors) < 3:
            # Perturb correct value
            jitter = sub_rng.choice([0.5, 1.5, 2.0, 3.0])
            if need_pi:
                fake_pi = total_pi + (sub_rng.randint(1, 6) * (1 if sub_rng.random() > 0.5 else -1))
                fake_int = total_int
                fake = _fmt_pair((fake_int, fake_pi))
            else:
                fake = f"{int(round((total_int + total_pi * math.pi) * jitter))}"
            if fake not in seen:
                distractors.append(fake)
                seen.add(fake)

        options = [correct_str] + distractors
        sub_rng.shuffle(options)
        correct_letter = "ABCD"[options.index(correct_str)]

        img = self._draw_compound(sub_rng, components, a, b, c)

        opt_str = "  ".join(f"({chr(65+i)}) {options[i]}" for i in range(4))
        op_note = ""
        if cfg["operation_type"] == "subtraction":
            op_note = " (Note: the inner component is scooped out of the main solid.)"
        elif cfg["operation_type"] == "subtraction_and_derived":
            op_note = (" (Note: the inner component is scooped out. Use the "
                       "diagram's labels to deduce any implied dimensions.)")

        if need_pi:
            fmt_note = " Express your answer with pi where needed."
        else:
            fmt_note = " Answer with a single integer."

        stems = [
            (f"The figure shows a compound 3D solid. What is its total volume?"
             f"{op_note}{fmt_note}\n{opt_str}\nAnswer with a single letter."),
            (f"Compute the total volume of the composite solid shown."
             f"{op_note}{fmt_note}\n{opt_str}\nAnswer with a single letter."),
        ]
        q = sub_rng.choice(stems)
        return q, correct_letter, img

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def _draw_compound(self, sub_rng, components, a, b, c) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fig = plt.figure(figsize=(7.5 * sc, 8 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(style["bg_color"])

        palette = list(style["palette"])

        # Draw base prism (first component assumed prism)
        base = components[0]
        kind0, params0, sign0, _, _ = base
        base_a = params0["a"]
        base_b = params0["b"]
        base_c = params0["c"]
        base_z = params0["z"]
        self._draw_prism(ax, base_a, base_b, base_c, 0, 0, base_z,
                         facecolor=palette[0], alpha=0.28)
        lfs = style["font_size_base"] + 3
        ax.text(base_a / 2, -1.4, base_z - 0.2, f"a={base_a}",
                fontsize=lfs, color="#b91c1c", fontweight="bold")
        ax.text(base_a + 0.8, base_b / 2, base_z - 0.2, f"b={base_b}",
                fontsize=lfs, color="#166534", fontweight="bold")
        ax.text(-1.5, -0.3, base_z + base_c / 2, f"c={base_c}",
                fontsize=lfs, color="#1e3a8a", fontweight="bold")

        # Draw remaining components
        z_top = base_z + base_c
        for i, (kind, params, sign, vi, vp) in enumerate(components[1:]):
            pc = palette[(i + 1) % len(palette)]
            if sign == -1:
                pc = "#888888"          # grey for subtracted
                al = 0.55
            else:
                al = 0.35
            sub_fs = style["font_size_base"] + 1
            if kind == "prism":
                self._draw_prism(ax, params["a"], params["b"], params["c"],
                                 params["x"], params["y"], params["z"],
                                 facecolor=pc, alpha=al,
                                 dashed=(sign == -1))
                ax.text(params["x"] + params["a"] + 0.4,
                        params["y"] + params["b"] + 0.4,
                        params["z"] + params["c"] + 0.2,
                        f"prism {params['a']}x{params['b']}x{params['c']}"
                        + (" (scooped)" if sign == -1 else ""),
                        fontsize=sub_fs,
                        color=pc if sign != -1 else "#444",
                        fontweight="bold")
            elif kind == "cylinder":
                self._draw_cylinder(ax, params["r"], params["h"],
                                    params["cx"], params["cy"], params["z"],
                                    color=pc, alpha=al,
                                    dashed=(sign == -1))
                ax.text(params["cx"] + params["r"] + 0.6,
                        params["cy"] + 0.4,
                        params["z"] + params["h"] + 0.3,
                        f"cyl r={params['r']}, h={params['h']}"
                        + (" (scooped)" if sign == -1 else ""),
                        fontsize=sub_fs,
                        color=pc if sign != -1 else "#444",
                        fontweight="bold")
            elif kind == "cone":
                self._draw_cone(ax, params["r"], params["h"],
                                params["cx"], params["cy"], params["z"],
                                color=pc, alpha=al,
                                dashed=(sign == -1))
                ax.text(params["cx"] + params["r"] + 0.6,
                        params["cy"] + 0.4,
                        params["z"] + params["h"] + 0.3,
                        f"cone r={params['r']}, h={params['h']}"
                        + (" (scooped)" if sign == -1 else ""),
                        fontsize=sub_fs,
                        color=pc if sign != -1 else "#444",
                        fontweight="bold")
            elif kind == "sphere":
                self._draw_hemisphere(ax, params["r"],
                                      params["cx"], params["cy"], params["z"],
                                      color=pc, alpha=al,
                                      dashed=(sign == -1))
                ax.text(params["cx"] + params["r"] + 0.6,
                        params["cy"] + 0.4,
                        params["z"] + params["r"] + 0.3,
                        f"hemi r={params['r']}"
                        + (" (scooped)" if sign == -1 else ""),
                        fontsize=sub_fs,
                        color=pc if sign != -1 else "#444",
                        fontweight="bold")
            if sign == +1:
                z_top = max(z_top, params.get("z", 0) + params.get(
                    "c", params.get("h", params.get("r", 0))))

        m = max(base_a, base_b, z_top + 1) * 1.1
        ax.set_xlim([-1, base_a + 2])
        ax.set_ylim([-1, base_b + 2])
        ax.set_zlim([0, z_top + 1.5])
        ax.view_init(elev=sub_rng.randint(18, 28),
                     azim=sub_rng.choice([30, 40, 50, 60]))
        ax.set_title(sub_rng.choice(_TITLES),
                     fontsize=style["font_size_base"] + 3,
                     fontweight="bold", pad=12)
        try:
            fig.tight_layout()
        except Exception:
            pass
        return self.fig_to_pil(fig, dpi=max(style["dpi"], 130))

    def _draw_prism(self, ax, a, b, c, x0, y0, z0,
                    facecolor="#4285f4", alpha=0.3, dashed=False):
        x1 = x0 + a
        y1 = y0 + b
        z1 = z0 + c
        verts = [
            [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],     # bottom
            [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],     # top
            [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],     # front
            [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],     # right
            [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],     # back
            [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],     # left
        ]
        ls = "dashed" if dashed else "solid"
        poly = Poly3DCollection(verts, alpha=alpha,
                                facecolors=[facecolor] * 6,
                                edgecolors="black", linewidths=1.3,
                                linestyles=ls)
        ax.add_collection3d(poly)

    def _draw_cylinder(self, ax, r, h, cx, cy, z0,
                       color="#1abc9c", alpha=0.35, dashed=False):
        theta = np.linspace(0, 2 * np.pi, 40)
        zs = np.linspace(z0, z0 + h, 2)
        theta_g, z_g = np.meshgrid(theta, zs)
        X = cx + r * np.cos(theta_g)
        Y = cy + r * np.sin(theta_g)
        ax.plot_surface(X, Y, z_g, alpha=alpha, color=color)
        ls = "dashed" if dashed else "solid"
        for zv in [z0, z0 + h]:
            ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), zv,
                    color="black", lw=1.3, linestyle=ls)

    def _draw_cone(self, ax, r, h, cx, cy, z0,
                   color="#e67e22", alpha=0.35, dashed=False):
        theta = np.linspace(0, 2 * np.pi, 40)
        zl = np.linspace(0, h, 12)
        theta_g, z_g = np.meshgrid(theta, zl)
        r_g = r * (1 - z_g / h)
        X = cx + r_g * np.cos(theta_g)
        Y = cy + r_g * np.sin(theta_g)
        Z = z_g + z0
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color)
        ls = "dashed" if dashed else "solid"
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), z0,
                color="black", lw=1.3, linestyle=ls)

    def _draw_hemisphere(self, ax, r, cx, cy, z0,
                         color="#9b59b6", alpha=0.35, dashed=False):
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi / 2, 18)
        X = cx + r * np.outer(np.cos(u), np.sin(v))
        Y = cy + r * np.outer(np.sin(u), np.sin(v))
        Z = z0 + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(X, Y, Z, alpha=alpha, color=color)
        ls = "dashed" if dashed else "solid"
        theta = np.linspace(0, 2 * np.pi, 40)
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), z0,
                color="black", lw=1.3, linestyle=ls)

if __name__ == "__main__":
    env = Compound3dVolumeQA()
    for lv in [0, 3, 6, 9]:
        gt = {}
        for seed in range(10):
            if env.generate(seed, {"level": lv}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{lv}: {gt}")
