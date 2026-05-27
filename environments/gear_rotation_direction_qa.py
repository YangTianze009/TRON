"""
Gear Rotation Direction QA (v4 G19).

Targets: spatial-vision task.MechanicalSystem -3.75.

Failure mode (from flipped cases idx=163/187): model collapses to "Answer: X"
without tracing the rotation direction through the chain. Avg pred length 16
chars — essentially no reasoning.

Task: render a gear train (2-4 gears connected in a chain or planetary
arrangement). Given the rotation direction of one gear, ask the resulting
direction of a target gear (or its revolution direction if planetary).

Reward: MCQ letter exact match. Require the model to enumerate
rotation direction through each gear interface.

Level axes:
  A) Number of gears: 2 at L0-2, 3 at L3-5, 4 at L6+
  B) Gear arrangement: linear chain at L0-4, add fixed-shaft + planetary at L5+
  C) Target quantity: rotation direction at L0-5, both rotation AND revolution at L6+
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

# 2026-05-05 R5 B3B HARDEN: stripped baked-in option strings from prompts.
# Options are now appended programmatically (5-MCQ A-E). Old templates carried
# fixed "A. clockwise B. counterclockwise C. does not rotate D. depends on
# load" wording inline — that prevented adding "E. No correct answer".
_TEMPLATES_CHAIN = [
    "A gear train with {n} meshing/linked gears is shown. Gear 1 rotates {input_dir}. What direction does gear {target} rotate? Trace each link.",
    "Given the gear chain, gear 1 spins {input_dir}. Determine gear {target}'s rotation direction. Trace each link.",
    "The figure shows {n} gears connected by various links. Gear 1 moves {input_dir}. What direction does gear {target} rotate?",
    "{n}-gear train. Gear 1: {input_dir}. Determine gear {target}'s rotation direction.",
    "In the gear train, gear 1 rotates {input_dir}. Gear {target} rotates in which direction?",
    "Gears connect in sequence. Input gear 1 direction: {input_dir}. Output gear {target}: which direction?",
    "A chain of {n} gears. Input gear rotates {input_dir}. Direction of gear {target}?",
    "Gear 1 ({input_dir}) is linked to a chain of {n} gears. What is the direction of gear {target}?",
    "Trace the gear chain. Gear 1 rotates {input_dir}. Determine gear {target}'s rotation.",
    "Gear chain ({n} gears). Gear 1: {input_dir}. Gear {target} direction?",
    "Given the {n}-gear chain with gear 1 rotating {input_dir}, determine gear {target}'s direction.",
    "The gear train has {n} gears. Input: gear 1 {input_dir}. What is gear {target}'s rotation direction?",
    "Starting gear 1 rotates {input_dir}. Trace through the gear chain to find gear {target}'s direction.",
    "{n} gears connect via various links. Gear 1 direction: {input_dir}. What is the direction of gear {target}?",
    "Each gear interacts via its link to the neighbour. Gear 1 {input_dir}. What direction does gear {target} rotate?",
    "The gear train in the figure has {n} gears. Gear 1 rotates {input_dir}. Determine gear {target}'s rotation direction.",
]

class GearRotationDirectionQA(StandaloneVisualEnv):
    ENV_NAME = "gear_rotation_direction"

    # 2026-05-04 R4: full-gradient redesign per mechanical-system sample.
    # Pure binary CW/CCW with mesh-only chain saturates by L4 because the
    # rule "alternates" trivializes. Real progressive gradient needs:
    #   - L0/L1: 2-3 gears mesh-only (trivial)
    #   - L2/L3: 4-5 gears mesh-only (still alternation but more counting)
    #   - L4/L5: mesh + belt (each link flips OR preserves based on type)
    #   - L6/L7: mesh + belt + IDLE COMPOUND (gears on shared shaft rotate
    #            in same direction even if one is "between" — break the
    #            naive alternation rule)
    #   - L8: 7 gears mesh + belt + compound shaft
    #   - L9: 9 gears all 3 link types (mesh, belt, compound) + question
    #         can also ask CCW/CW of arbitrary mid-chain gear
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-05 R5 B3B HARDEN:
        #   - All levels now use 5-MCQ {CW, CCW, "Doesn't rotate",
        #     "Cannot determine", "No correct answer"}
        #   - L8/L9 add "locked gear" scenario: one mid-chain gear is FROZEN.
        #     Chain transmits motion only up to the lock; downstream gears
        #     (including target) "do not rotate". 50% trigger probability.
        #   - L8/L9 e_trap rate 40% (all 4 visible options wrong → answer E).
        if level == 0:
            return {"n_gears": 2, "level": level,
                    "link_types_pool": ["mesh"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 1:
            return {"n_gears": 3, "level": level,
                    "link_types_pool": ["mesh"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 2:
            return {"n_gears": 4, "level": level,
                    "link_types_pool": ["mesh"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 3:
            return {"n_gears": 5, "level": level,
                    "link_types_pool": ["mesh"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 4:
            return {"n_gears": 4, "level": level,
                    "link_types_pool": ["mesh", "belt"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 5:
            return {"n_gears": 5, "level": level,
                    "link_types_pool": ["mesh", "belt"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 6:
            return {"n_gears": 5, "level": level,
                    "link_types_pool": ["mesh", "belt", "compound"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 7:
            return {"n_gears": 6, "level": level,
                    "link_types_pool": ["mesh", "belt", "compound"],
                    "locked_chance": 0.0, "e_trap_rate": 0.0}
        if level == 8:
            return {"n_gears": 7, "level": level,
                    "link_types_pool": ["mesh", "belt", "compound"],
                    "locked_chance": 0.50, "e_trap_rate": 0.40}
        return {"n_gears": 9, "level": level,
                "link_types_pool": ["mesh", "belt", "compound"],
                "locked_chance": 0.50, "e_trap_rate": 0.40}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 871)
        self._primary_complexity_feature = level

        n = cfg["n_gears"]
        input_dir = rng.choice(["clockwise", "counterclockwise"])
        target = rng.randint(2, n)
        rng_link = random.Random((self.seed or 0) * 7 + level)
        link_pool = cfg["link_types_pool"]
        link_types = [rng_link.choice(link_pool) for _ in range(n - 1)]

        # 2026-05-05 R5 B3B HARDEN: locked-gear scenario at L8/L9.
        # Pick a random gear in 2..n-1 (not gear 1, not target) and freeze it.
        # If the locked gear is at index k (1-indexed), every gear k..n cannot
        # rotate (the chain jams). If target >= k, answer = "does not rotate".
        locked_gear = None
        if cfg.get("locked_chance", 0.0) > 0.0 and rng.random() < cfg["locked_chance"]:
            # Lock somewhere between gear 2 and gear (n-1). Must be ≤ target
            # to actually affect the target's rotation; otherwise the lock
            # is downstream of target and doesn't matter — pick at most target.
            lock_max = max(2, target)
            lock_min = 2
            if lock_max >= lock_min:
                locked_gear = rng.randint(lock_min, lock_max)
        self._locked_gear = locked_gear

        # Compute target direction by tracing each link
        # mesh: flips direction; belt: preserves direction; compound (shared
        # shaft): preserves direction (rigid coupling).
        cur_dir = input_dir
        jammed = False
        for i in range(target - 1):
            # i is 0-indexed; link i connects gear (i+1) -> gear (i+2)
            if locked_gear is not None and (i + 2) >= locked_gear:
                # Once we cross the locked gear, the chain stops.
                jammed = True
                break
            if link_types[i] == "mesh":
                cur_dir = ("counterclockwise" if cur_dir == "clockwise"
                           else "clockwise")
            # "belt" or "compound": no flip
        if jammed:
            target_dir = "does not rotate"
        else:
            target_dir = cur_dir
        self._link_types = link_types

        # 5-MCQ option set (was 4-MCQ in R4)
        # 2026-05-05 R5 B3B HARDEN: replaced "depends on load" → "Cannot be
        # determined from the diagram" + added E "No correct answer".
        OPTIONS_5 = ["clockwise", "counterclockwise", "does not rotate",
                     "Cannot be determined from the diagram",
                     "No correct answer"]

        # E-trap branch: if rng < e_trap_rate, replace correct value in the
        # visible options with a *different* wrong direction, set GT='E'.
        e_trap_rate = cfg.get("e_trap_rate", 0.0)
        use_etrap = (rng.random() < e_trap_rate) if e_trap_rate > 0.0 else False

        if use_etrap:
            # Build 4 visible options that all are WRONG. Pick the wrong-but-
            # plausible distractors that are NOT target_dir.
            wrong_pool = [o for o in OPTIONS_5[:4] if o != target_dir]
            # If target_dir is one of the first 4, we now have 3 wrongs;
            # need 4 total visible. Pad by reusing 'Cannot be determined' if
            # already present, else add it; else reuse 'does not rotate'.
            visible = wrong_pool[:4]
            if len(visible) < 4:
                # extend with whichever of the first 4 is missing
                for o in OPTIONS_5[:4]:
                    if o not in visible:
                        visible.append(o)
                    if len(visible) >= 4:
                        break
            rng.shuffle(visible)
            visible_4 = visible[:4]
            answer_letter = "E"
            options_text = visible_4 + ["No correct answer"]
        else:
            # Normal: 5 fixed slots A=CW B=CCW C=no-rot D=cannot-determine E=no-correct
            options_text = OPTIONS_5
            try:
                correct_idx = options_text.index(target_dir)
            except ValueError:
                # target_dir not in options (shouldn't happen) — fall back
                correct_idx = 0
            answer_letter = "ABCDE"[correct_idx]

        sidx = (self.seed or 0) % len(_TEMPLATES_CHAIN)
        stem = _TEMPLATES_CHAIN[sidx].format(n=n, input_dir=input_dir,
                                              target=target)

        # Rules block (so the model has the link semantics). Always describe
        # link types because mesh-only L0/L1 still benefits.
        link_desc = " ".join(
            f"Gear{i+1}-Gear{i+2}: {link_types[i]}." for i in range(n - 1)
        )
        rule_pieces = ["mesh flips direction"]
        if "belt" in link_pool:
            rule_pieces.append("belt preserves direction")
        if "compound" in link_pool:
            rule_pieces.append("compound (same shaft) preserves direction")
        if locked_gear is not None:
            rule_pieces.append("a LOCKED gear stops the entire chain past it")
        rules = "; ".join(rule_pieces) + "."

        opts_block = "\n".join(
            f"{chr(ord('A')+i)}. {options_text[i]}" for i in range(5))
        lock_text = (f" Note: gear {locked_gear} is LOCKED (frozen)."
                     if locked_gear is not None else "")
        question = (
            f"{stem}{lock_text} Link types: {link_desc} ({rules})\n\n"
            f"{opts_block}\n\nChoose the correct option (A, B, C, D, or E)."
        )

        img = self._render(n, input_dir, target, rng,
                           locked_gear=locked_gear)
        return question, answer_letter, img

    def _render(self, n, input_dir, target, rng, locked_gear=None):
        fig, ax = plt.subplots(figsize=(3 + n * 1.5, 4))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1, 1.5 + n * 1.8)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.axis("off")

        # Draw gears as circles with teeth
        for i in range(n):
            cx = 0.8 + i * 1.6
            cy = 0
            r = 0.7
            gear_idx = i + 1
            is_locked = (locked_gear is not None and gear_idx == locked_gear)
            # Locked gear: red fill instead of grey
            face = "#ff8888" if is_locked else "#cccccc"
            ax.add_patch(mpatches.Circle((cx, cy), r, fc=face,
                                          ec="black", lw=1.5))
            # Teeth (simple tick marks)
            for theta_deg in range(0, 360, 30):
                theta = math.radians(theta_deg)
                xa = cx + r * math.cos(theta)
                ya = cy + r * math.sin(theta)
                xb = cx + (r + 0.12) * math.cos(theta)
                yb = cy + (r + 0.12) * math.sin(theta)
                ax.plot([xa, xb], [ya, yb], color="black", lw=1.2)
            # Center hole
            ax.add_patch(mpatches.Circle((cx, cy), 0.12, fc="#444444",
                                          ec="black", lw=1.0))
            # Gear number
            ax.text(cx, -0.9, str(gear_idx), fontsize=14, ha="center",
                    fontweight="bold")
            # Locked badge
            if is_locked:
                ax.text(cx, cy + 0.05, "LOCK", fontsize=9, ha="center",
                        color="darkred", fontweight="bold")

        # Draw direction arrow on gear 1
        cx1, cy1 = 0.8, 0
        r = 0.45
        if input_dir == "clockwise":
            arc = mpatches.FancyArrow(cx1, cy1 + r, 0.35, -0.15,
                                        width=0.04, color="red")
            # simple text
            ax.text(cx1, cy1 + r + 0.35, "CW" if input_dir == "clockwise" else "CCW",
                    fontsize=11, ha="center", color="red", fontweight="bold")
        else:
            ax.text(cx1, cy1 + r + 0.35, "CCW",
                    fontsize=11, ha="center", color="red", fontweight="bold")
        # Mark target with "?"
        cxt = 0.8 + (target - 1) * 1.6
        ax.text(cxt, 0 + r + 0.35, "?", fontsize=16, ha="center",
                color="darkgreen", fontweight="bold")

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_grd"
    os.makedirs(out_dir, exist_ok=True)
    env = GearRotationDirectionQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 801
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[grd L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/grd_s{s}_L{level}.png")
            print(f"[grd L{level} s{s}] A={env._answer}")
