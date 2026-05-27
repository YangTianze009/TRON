"""
Mechanical-aptitude MCQ family. Each problem instance picks one of several
sub-templates from a Bennett-style mechanical comprehension test:

  * gear-direction-prediction (2-4 meshed gears)
  * lever-pole-pressure-comparison
  * weight-balance scale
  * friction-motion-classify (sled runner width)
  * pulley-load-compare (with/without passive wheel)
  * parallel-resistance-compute
  * series-resistance-compute
  * hooke-spring stretch
  * boyle-piston-pressure
  * inclined-plane-effort

Single-letter MCQ answer (A/B/C/D).

Visual style: clean monochrome line diagrams with labelled parts;
gears as serrated circles with an arrow on the driver, levers as
horizontal bars on triangular fulcrums, springs as zig-zag lines,
circuits as resistor symbols (rectangles) on lines.
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


_SUB_TEMPLATES = [
    # Originals (10)
    "gear_direction",
    "weight_balance",
    "sled_runner",
    "pulley_lift",
    "series_resistance",
    "parallel_resistance",
    "hooke_spring",
    "boyle_piston",
    "inclined_plane",
    "lever_balance_distance",
    # New: common-sense mechanical (3-candidate row of pictograms)
    "bird_drag",
    "drone_stability",
    "parcel_pivot",
    "halligan_lever",
    "spade_carry",
    "balloon_pressure",
    "granary_volume",
    "helicopter_tilt",
    "candle_jar",
    "pole_pressure_compare",
    # New: physical-intuition diagrams
    "torricelli_outflow",
    "magnifying_glass",
    "bridge_deflection",
    "convection_house",
    "convex_mirror",
    "fire_door_spring",
    "fire_engine_speed",
    "cylinder_underwater",
    "pendulum_compare",
    "centripetal_string",
    # New: singleton template additions
    "belt_rpm_statement",
    "rack_pinion",
    "seesaw_acrobat",
    "capacitor_id",
    "balls_falling",
    "pendulum_wagon",
    "bolt_cutter",
]


# Per-template question stems (≥16 templates total across the family). For each
# sub-template we have at least 2-4 wordings; they're sampled per-seed.

_GEAR_TEMPLATES = [
    "In which direction does the orange gear rotate? (A) Clockwise (B) Counterclockwise (C) No rotation. Reply with the letter (A/B/C) in <answer>...</answer>.",
    "Looking at the gear arrangement, in which direction will the orange gear turn? (A) Clockwise (B) Counterclockwise (C) Does not rotate. Letter (A/B/C) in <answer>...</answer>.",
    "Examine the meshed gears in the diagram. The driver gear turns as shown by the arrow. In which direction does the orange gear rotate? (A) Clockwise (B) Counterclockwise (C) No rotation. Letter in <answer>...</answer>.",
    "The arrow on the driver gear indicates its direction of rotation. In which direction does the orange gear rotate? (A) Clockwise (B) Counterclockwise (C) No rotation. Letter in <answer>...</answer>.",
]

_BALANCE_TEMPLATES = [
    "The scale is balanced. What is the weight of the unknown weight (marked '?'): (A) {a} {u} (B) {b} {u} (C) {c} {u} (D) {d} {u}? Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "A balanced lever-and-fulcrum scale is shown. Find the unknown mass: (A) {a} {u} (B) {b} {u} (C) {c} {u} (D) {d} {u}. Letter (A/B/C/D) in <answer>...</answer>.",
    "The lever scale is in equilibrium. What weight does the question mark represent? (A) {a} {u} (B) {b} {u} (C) {c} {u} (D) {d} {u}. Letter in <answer>...</answer>.",
]

_SLED_TEMPLATES = [
    "Considering only the width of the runners, which sled is most likely the fastest down an icy slope? (A) Sled A (B) Sled B (C) Sled C. Reply with the letter (A/B/C) in <answer>...</answer>.",
    "Three sleds with different runner widths are shown. Which one will go fastest down an icy slope, considering runner width only? (A) Sled A (B) Sled B (C) Sled C. Letter in <answer>...</answer>.",
    "Below are three sleds. Friction increases with runner contact area. Which sled is fastest down an icy slope? (A) A (B) B (C) C. Letter (A/B/C) in <answer>...</answer>.",
]

_PULLEY_TEMPLATES = [
    "Two pulley arrangements are shown. In which case will less effort be needed to lift the weight? (A) Arrangement A (B) Arrangement B (C) Equal effort. Reply with the letter (A/B/C) in <answer>...</answer>.",
    "Compare the two pulley setups. Which arrangement requires less force to lift the load? (A) A (B) B (C) Equal. Letter (A/B/C) in <answer>...</answer>.",
    "Two pulley systems lift the same weight. Which one needs less force from the operator? (A) A (B) B (C) The same. Letter in <answer>...</answer>.",
]

_SERIES_R_TEMPLATES = [
    "What is the total resistance of the circuit if R1 = {r1} ohm and R2 = {r2} ohm in series? (A) {a} ohm (B) {b} ohm (C) {c} ohm (D) {d} ohm. Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "In the circuit shown, two resistors are wired in series. With R1 = {r1} ohm and R2 = {r2} ohm, what is the total resistance? (A) {a} (B) {b} (C) {c} (D) {d} ohm. Letter (A/B/C/D) in <answer>...</answer>.",
    "Two resistors are connected in series. R1 = {r1} ohm and R2 = {r2} ohm. Total resistance? (A) {a} (B) {b} (C) {c} (D) {d}. Letter in <answer>...</answer>.",
]

_PARALLEL_R_TEMPLATES = [
    "What is the total resistance of the circuit shown, with the two resistors in parallel? R1 = {r1} ohm, R2 = {r2} ohm. (A) {a} (B) {b} (C) {c} (D) {d} ohm. Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "The two resistors in the circuit are wired in parallel. R1 = {r1} ohm, R2 = {r2} ohm. Total resistance? (A) {a} (B) {b} (C) {c} (D) {d}. Letter in <answer>...</answer>.",
    "Compute the total resistance of the parallel circuit shown (R1 = {r1}, R2 = {r2}). (A) {a} (B) {b} (C) {c} (D) {d} ohm. Letter (A/B/C/D) in <answer>...</answer>.",
]

_HOOKE_TEMPLATES = [
    "A spring stretches {x1} {ux} when a {w1} {uw} weight hangs from it. If the weight is increased to {w2} {uw}, how far will the spring stretch? Assume Hooke's law. (A) {a} {ux} (B) {b} {ux} (C) {c} {ux} (D) {d} {ux}. Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Hooke's law applies. A {w1} {uw} weight stretches the spring {x1} {ux}. How much will the spring stretch with a {w2} {uw} weight? (A) {a} (B) {b} (C) {c} (D) {d} {ux}. Letter (A/B/C/D) in <answer>...</answer>.",
]

_BOYLE_TEMPLATES = [
    "A sealed cylinder full of gas has a piston that moves from position A to position B, reducing the volume to {frac} of the original. What happens to the pressure? (A) increases by a factor of {a} (B) increases by a factor of {b} (C) stays the same (D) decreases by a factor of {a}. Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "A piston compresses gas in a sealed cylinder. The new volume is {frac} of the original. What is the new pressure? (A) {a}x the original (B) {b}x the original (C) the same (D) 1/{a}x the original. Letter (A/B/C/D) in <answer>...</answer>.",
]

_INCLINED_TEMPLATES = [
    "A {w} {uw} ball must be pushed up to a platform {h} {ul} above the ground. A {L} {ul} long ramp is available. How much effort (in {uw}) is required, ignoring friction? (A) {a} (B) {b} (C) {c} (D) {d}. Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Using a {L} {ul} ramp to push a {w} {uw} object onto a platform {h} {ul} above the ground, what minimum force (in {uw}) is needed? (A) {a} (B) {b} (C) {c} (D) {d}. Letter (A/B/C/D) in <answer>...</answer>.",
]

_LEVER_DIST_TEMPLATES = [
    "A lever balances at the fulcrum. Mass M1 = {m1} {uw} is on the left and mass M2 = {m2} {uw} is on the right. The total distance between the two masses is {dtot} {ul}. At what distance from the left mass should the fulcrum be placed to balance the lever? (A) {a} (B) {b} (C) {c} (D) {d} {ul}. Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Two masses sit on a lever {dtot} {ul} apart. Mass M1 = {m1} {uw} (left) and M2 = {m2} {uw} (right). At what distance from the left should the fulcrum be placed to balance? (A) {a} (B) {b} (C) {c} (D) {d} {ul}. Letter (A/B/C/D) in <answer>...</answer>.",
]

_HINT_GEAR = " Hint: meshed gears rotate in opposite directions. Each pair flips the rotation."
_HINT_SERIES = " Hint: in series, total = R1 + R2."
_HINT_PARALLEL = " Hint: in parallel, 1/Rtot = 1/R1 + 1/R2."
_HINT_HOOKE = " Hint: stretch is proportional to the force."
_HINT_BOYLE = " Hint: at constant temperature, pressure is inversely proportional to volume."
_HINT_INCLINED = " Hint: effort = weight * (height / ramp length)."
_HINT_LEVER = " Hint: M1 * d1 = M2 * d2; d1 + d2 = dtot."
_HINT_BALANCE = " Hint: the torques on each side of the fulcrum balance."


class BennettMechanicalQA(StandaloneVisualEnv):
    ENV_NAME = "bennett_mechanical"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0-L1: only the simplest sub-templates (single-step, common-sense
        # picks-from-three pictograms)
        if level <= 1:
            return {
                "subs": [
                    "gear_direction", "sled_runner", "pulley_lift",
                    "bird_drag", "spade_carry", "drone_stability",
                    "balloon_pressure", "granary_volume",
                    "convex_mirror", "fire_engine_speed",
                ],
                "n_gears": (2, 2),
                "round_nums": True,
                "give_hint": True,
            }
        if level <= 3:
            return {
                "subs": [
                    "gear_direction", "weight_balance", "sled_runner",
                    "pulley_lift", "series_resistance", "hooke_spring",
                    "bird_drag", "drone_stability", "halligan_lever",
                    "spade_carry", "balloon_pressure", "granary_volume",
                    "candle_jar", "bridge_deflection", "convex_mirror",
                    "fire_door_spring", "fire_engine_speed",
                    "cylinder_underwater", "convection_house",
                    "pendulum_compare", "centripetal_string",
                    "capacitor_id",
                ],
                "n_gears": (2, 3),
                "round_nums": True,
                "give_hint": True,
            }
        if level <= 5:
            return {
                "subs": [
                    "gear_direction", "weight_balance", "sled_runner",
                    "pulley_lift", "series_resistance", "parallel_resistance",
                    "hooke_spring", "boyle_piston", "inclined_plane",
                    "lever_balance_distance",
                    "bird_drag", "drone_stability", "parcel_pivot",
                    "halligan_lever", "spade_carry", "balloon_pressure",
                    "granary_volume", "helicopter_tilt", "candle_jar",
                    "pole_pressure_compare",
                    "torricelli_outflow", "magnifying_glass",
                    "bridge_deflection", "convection_house",
                    "convex_mirror", "fire_door_spring",
                    "fire_engine_speed", "cylinder_underwater",
                    "pendulum_compare", "centripetal_string",
                    "rack_pinion", "seesaw_acrobat", "capacitor_id",
                    "balls_falling", "pendulum_wagon", "bolt_cutter",
                ],
                "n_gears": (2, 4),
                "round_nums": True,
                "give_hint": True,
            }
        if level <= 7:
            return {
                "subs": _SUB_TEMPLATES,
                "n_gears": (3, 4),
                "round_nums": True,
                "give_hint": False,
            }
        return {
            "subs": _SUB_TEMPLATES,
            "n_gears": (3, 5),
            "round_nums": False,
            "give_hint": False,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4421 + level * 71 + 23)

        # Pick a sub-template
        sub = rng.choice(cfg["subs"])

        for attempt in range(8):
            try:
                if sub == "gear_direction":
                    return self._gen_gear(rng, cfg, level)
                if sub == "weight_balance":
                    return self._gen_balance(rng, cfg, level)
                if sub == "sled_runner":
                    return self._gen_sled(rng, cfg, level)
                if sub == "pulley_lift":
                    return self._gen_pulley(rng, cfg, level)
                if sub == "series_resistance":
                    return self._gen_series_r(rng, cfg, level)
                if sub == "parallel_resistance":
                    return self._gen_parallel_r(rng, cfg, level)
                if sub == "hooke_spring":
                    return self._gen_hooke(rng, cfg, level)
                if sub == "boyle_piston":
                    return self._gen_boyle(rng, cfg, level)
                if sub == "inclined_plane":
                    return self._gen_inclined(rng, cfg, level)
                if sub == "lever_balance_distance":
                    return self._gen_lever_dist(rng, cfg, level)
                # ----- common-sense pictograms -----
                if sub == "bird_drag":
                    return self._gen_bird_drag(rng, cfg, level)
                if sub == "drone_stability":
                    return self._gen_drone_stability(rng, cfg, level)
                if sub == "parcel_pivot":
                    return self._gen_parcel_pivot(rng, cfg, level)
                if sub == "halligan_lever":
                    return self._gen_halligan_lever(rng, cfg, level)
                if sub == "spade_carry":
                    return self._gen_spade_carry(rng, cfg, level)
                if sub == "balloon_pressure":
                    return self._gen_balloon_pressure(rng, cfg, level)
                if sub == "granary_volume":
                    return self._gen_granary_volume(rng, cfg, level)
                if sub == "helicopter_tilt":
                    return self._gen_helicopter_tilt(rng, cfg, level)
                if sub == "candle_jar":
                    return self._gen_candle_jar(rng, cfg, level)
                if sub == "pole_pressure_compare":
                    return self._gen_pole_pressure_compare(rng, cfg, level)
                # ----- physical-intuition diagrams -----
                if sub == "torricelli_outflow":
                    return self._gen_torricelli_outflow(rng, cfg, level)
                if sub == "magnifying_glass":
                    return self._gen_magnifying_glass(rng, cfg, level)
                if sub == "bridge_deflection":
                    return self._gen_bridge_deflection(rng, cfg, level)
                if sub == "convection_house":
                    return self._gen_convection_house(rng, cfg, level)
                if sub == "convex_mirror":
                    return self._gen_convex_mirror(rng, cfg, level)
                if sub == "fire_door_spring":
                    return self._gen_fire_door_spring(rng, cfg, level)
                if sub == "fire_engine_speed":
                    return self._gen_fire_engine_speed(rng, cfg, level)
                if sub == "cylinder_underwater":
                    return self._gen_cylinder_underwater(rng, cfg, level)
                if sub == "pendulum_compare":
                    return self._gen_pendulum_compare(rng, cfg, level)
                if sub == "centripetal_string":
                    return self._gen_centripetal_string(rng, cfg, level)
                # ----- singleton template additions -----
                if sub == "belt_rpm_statement":
                    return self._gen_belt_rpm_statement(rng, cfg, level)
                if sub == "rack_pinion":
                    return self._gen_rack_pinion(rng, cfg, level)
                if sub == "seesaw_acrobat":
                    return self._gen_seesaw_acrobat(rng, cfg, level)
                if sub == "capacitor_id":
                    return self._gen_capacitor_id(rng, cfg, level)
                if sub == "balls_falling":
                    return self._gen_balls_falling(rng, cfg, level)
                if sub == "pendulum_wagon":
                    return self._gen_pendulum_wagon(rng, cfg, level)
                if sub == "bolt_cutter":
                    return self._gen_bolt_cutter(rng, cfg, level)
            except Exception:
                # fallback: try a different sub on next attempt
                sub = rng.choice(cfg["subs"])
                continue
        return None

    # =========================================================== #
    # Sub-template generators
    # =========================================================== #

    def _gen_gear(self, rng, cfg, level):
        n = rng.randint(*cfg["n_gears"])
        # Driver gear at index 0 rotates clockwise (CW). Each meshed gear flips.
        driver_dir = rng.choice([+1, -1])  # +1 = CW
        # Marked (orange) gear is the last in the chain
        marked_idx = n - 1
        # Direction of marked gear: each pairing flips
        marked_dir = driver_dir * ((-1) ** marked_idx)
        # Letter mapping: A=CW, B=CCW, C=no rotation
        if marked_dir == +1:
            answer = "A"
        else:
            answer = "B"
        q = rng.choice(_GEAR_TEMPLATES)
        if cfg["give_hint"]:
            q = q + _HINT_GEAR
        img = self._render_gears(n, driver_dir, marked_idx, rng)
        return q, answer, img

    def _gen_balance(self, rng, cfg, level):
        # Class-1 lever: known weight K at distance dK from fulcrum,
        # unknown weight ?  at distance dU from fulcrum on the other side.
        # K * dK = ?  * dU
        K = rng.choice([2, 4, 5, 6, 8, 10])
        dK = rng.choice([2, 3, 4, 5, 6])
        dU = rng.choice([1, 2, 3, 4])
        unknown = (K * dK) / dU
        if not float(unknown).is_integer():
            # try again
            for _ in range(20):
                K = rng.choice([2, 4, 5, 6, 8, 10])
                dK = rng.choice([2, 3, 4, 5, 6])
                dU = rng.choice([1, 2, 3, 4])
                if (K * dK) % dU == 0:
                    unknown = (K * dK) // dU
                    break
            else:
                unknown = K * dK // max(1, dU)
        unknown = int(unknown)
        # Build distractors
        opts_set = {unknown, K, K * dK, max(1, unknown // 2),
                    unknown + dK, unknown + K, max(1, unknown - 1)}
        opts_set.discard(unknown)
        distractors = list(opts_set)[:3]
        all_opts = [unknown] + distractors[:3]
        while len(all_opts) < 4:
            d = unknown + rng.choice([-3, -2, -1, 1, 2, 3, 5])
            if d > 0 and d not in all_opts:
                all_opts.append(d)
        rng.shuffle(all_opts)
        ans_idx = all_opts.index(unknown)
        answer = chr(ord("A") + ans_idx)
        u = "lb"
        q = rng.choice(_BALANCE_TEMPLATES).format(
            a=all_opts[0], b=all_opts[1], c=all_opts[2], d=all_opts[3], u=u)
        if cfg["give_hint"]:
            q = q + _HINT_BALANCE
        img = self._render_balance(K, dK, unknown, dU, u, rng)
        return q, answer, img

    def _gen_sled(self, rng, cfg, level):
        # Three sleds with different runner widths — narrowest is fastest
        widths = sorted([round(rng.uniform(0.4, 1.5), 1) for _ in range(3)])
        # Make sure they're distinguishable
        for _ in range(20):
            if widths[2] / widths[0] >= 1.6:
                break
            widths = sorted([round(rng.uniform(0.4, 1.5), 1) for _ in range(3)])
        # Random label assignment
        order = list(range(3))
        rng.shuffle(order)
        widths_for_letter = [widths[i] for i in order]
        narrowest_idx = widths_for_letter.index(min(widths_for_letter))
        answer = chr(ord("A") + narrowest_idx)
        q = rng.choice(_SLED_TEMPLATES)
        if cfg["give_hint"]:
            q = q + " Hint: thinner runners give less drag and friction on ice."
        img = self._render_sleds(widths_for_letter, rng)
        return q, answer, img

    def _gen_pulley(self, rng, cfg, level):
        # Two arrangements: A is single fixed pulley (effort = full weight).
        # B is single movable pulley (effort = weight / 2).
        # Sometimes flip to vary the answer
        # Coin flip: which letter has the movable (advantage) pulley
        movable_letter = rng.choice(["A", "B"])
        if movable_letter == "A":
            answer = "A"
        else:
            answer = "B"
        q = rng.choice(_PULLEY_TEMPLATES)
        if cfg["give_hint"]:
            q = q + " Hint: a movable (passive) pulley halves the required force; a fixed pulley does not."
        img = self._render_pulley(movable_letter, rng)
        return q, answer, img

    def _gen_series_r(self, rng, cfg, level):
        r1 = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])
        r2 = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])
        total = r1 + r2
        # Distractors: r1, r2, r1*r2, parallel_value rounded
        parallel = round((r1 * r2) / (r1 + r2), 1)
        opts_set = {total, r1, r2, r1 * r2, int(parallel) if float(parallel).is_integer() else round(parallel)}
        opts_set.discard(total)
        opts = [total] + list(opts_set)[:3]
        while len(opts) < 4:
            d = total + rng.choice([-3, -2, -1, 1, 2, 3, 4])
            if d > 0 and d not in opts:
                opts.append(d)
        rng.shuffle(opts)
        ans_idx = opts.index(total)
        answer = chr(ord("A") + ans_idx)
        q = rng.choice(_SERIES_R_TEMPLATES).format(
            r1=r1, r2=r2, a=opts[0], b=opts[1], c=opts[2], d=opts[3])
        if cfg["give_hint"]:
            q = q + _HINT_SERIES
        img = self._render_circuit("series", r1, r2, rng)
        return q, answer, img

    def _gen_parallel_r(self, rng, cfg, level):
        # Pick r1, r2 such that the parallel result is nice (an integer or .5)
        for _ in range(40):
            r1 = rng.choice([2, 3, 4, 6, 8, 10, 12, 15, 20, 24])
            r2 = rng.choice([2, 3, 4, 6, 8, 10, 12, 15, 20, 24])
            total = (r1 * r2) / (r1 + r2)
            # Accept if it's at least a multiple of 0.5
            if abs(total * 2 - round(total * 2)) < 1e-6:
                total_disp = round(total, 1)
                break
        else:
            r1, r2 = 6, 12
            total_disp = round((r1 * r2) / (r1 + r2), 1)
        # Distractors
        series_val = r1 + r2
        opts_set = {total_disp, r1, r2, series_val, round(total_disp + 1, 1),
                    round(total_disp + 2, 1)}
        opts_set.discard(total_disp)
        opts = [total_disp] + list(opts_set)[:3]
        while len(opts) < 4:
            d = round(total_disp + rng.choice([-1.5, -1.0, 1.0, 1.5, 2.0]), 1)
            if d > 0 and d not in opts:
                opts.append(d)
        rng.shuffle(opts)
        ans_idx = opts.index(total_disp)
        answer = chr(ord("A") + ans_idx)
        # Format display: drop ".0" for integers
        def _fmt(x):
            return str(int(x)) if float(x).is_integer() else str(x)
        q = rng.choice(_PARALLEL_R_TEMPLATES).format(
            r1=r1, r2=r2,
            a=_fmt(opts[0]), b=_fmt(opts[1]),
            c=_fmt(opts[2]), d=_fmt(opts[3]))
        if cfg["give_hint"]:
            q = q + _HINT_PARALLEL
        img = self._render_circuit("parallel", r1, r2, rng)
        return q, answer, img

    def _gen_hooke(self, rng, cfg, level):
        # Pick (W1, x1) and (W2, x2 = x1 * W2 / W1) so that x2 is nice
        for _ in range(40):
            W1 = rng.choice([5, 10, 15, 20, 25, 30])
            W2 = rng.choice([10, 15, 20, 25, 30, 40, 50])
            if W2 == W1:
                continue
            x1_int = rng.choice([1, 2, 3, 4, 5])
            x1 = x1_int * 0.5  # in inches: 0.5, 1.0, ...
            x2 = x1 * W2 / W1
            # Accept if x2 is a multiple of 0.25 and < 4
            if abs(x2 * 4 - round(x2 * 4)) < 1e-6 and 0 < x2 < 4 and x2 != x1:
                x2 = round(x2, 2)
                break
        else:
            W1, W2, x1, x2 = 10, 15, 0.5, 0.75

        def _fmt(x):
            return str(int(x)) if float(x).is_integer() else str(x)
        # Distractors
        opts_set = {x2, x1, round(x1 + (x2 - x1) * 2, 2),
                    round(x2 / 2, 2), round(x1 * 2, 2),
                    round(x1 + 0.25, 2)}
        opts_set.discard(x2)
        opts = [x2] + [v for v in opts_set if v > 0][:3]
        while len(opts) < 4:
            d = round(x2 + rng.choice([-0.5, -0.25, 0.25, 0.5, 0.75]), 2)
            if d > 0 and d not in opts:
                opts.append(d)
        rng.shuffle(opts)
        ans_idx = opts.index(x2)
        answer = chr(ord("A") + ans_idx)
        ux, uw = "in", "lb"
        q = rng.choice(_HOOKE_TEMPLATES).format(
            x1=_fmt(x1), w1=W1, w2=W2, ux=ux, uw=uw,
            a=_fmt(opts[0]), b=_fmt(opts[1]),
            c=_fmt(opts[2]), d=_fmt(opts[3]))
        if cfg["give_hint"]:
            q = q + _HINT_HOOKE
        img = self._render_spring(W1, x1, ux, uw, rng)
        return q, answer, img

    def _gen_boyle(self, rng, cfg, level):
        # Pick a compression factor (e.g. volume reduces to 1/n)
        n = rng.choice([2, 3, 4])
        factor = n  # pressure increases by n
        frac_str = f"1/{n}"
        # Options: n, n+1, n-1, the same
        opts_text = [f"{n}", f"{n + 1}", f"{n - 1}" if n > 1 else "5"]
        # Build choices A/B/C/D as integers
        a = n
        b = n + 1
        # Make sure all 4 unique
        opts_int = [n, n + 1, max(1, n - 1), n + 2]
        # Letter A: increases by `a`, B: increases by `b`, C: stays the same, D: decreases by 1/`a`
        # Correct = letter A
        rng_letters = list("ABCD")
        # Shuffle which letter holds the correct option
        # Re-using literal text here makes randomization complicated; instead,
        # rotate the options so the correct one lands in different positions.
        # We'll use a different formulation: just place the 4 options as factors
        # and the question hands them as (A) increases by F1 (B) increases by F2 (C) stays (D) decreases by F1.
        # Cleanest: have A/B as unique integers; correct is always the value `n`.
        # To vary the correct letter, swap a and b.
        if rng.random() < 0.5:
            a_val, b_val = n, n + 1
            answer = "A"
        else:
            a_val, b_val = n + 1, n
            answer = "B"
        q = rng.choice(_BOYLE_TEMPLATES).format(
            frac=frac_str, a=a_val, b=b_val)
        if cfg["give_hint"]:
            q = q + _HINT_BOYLE
        img = self._render_piston(n, rng)
        return q, answer, img

    def _gen_inclined(self, rng, cfg, level):
        # Effort = W * h / L (frictionless inclined plane).
        # Pick W, h, L so that effort is integer.
        for _ in range(60):
            W = rng.choice([60, 90, 120, 150, 180, 200, 240, 270, 300])
            ratio_h_L = rng.choice([(1, 3), (1, 4), (1, 5), (1, 6), (2, 5), (2, 7), (3, 10)])
            h_unit, L_unit = ratio_h_L
            h_mul = rng.choice([3, 4, 5, 6, 8])
            h, L = h_unit * h_mul, L_unit * h_mul
            effort = W * h / L
            if float(effort).is_integer() and effort > 0 and effort < W:
                effort = int(effort)
                break
        else:
            W, h, L, effort = 180, 15, 45, 60
        opts_set = {effort, W, W // 2, W - effort, effort + 15, effort - 15, h * 4}
        opts_set.discard(effort)
        opts = [effort] + [v for v in opts_set if v > 0][:3]
        while len(opts) < 4:
            d = effort + rng.choice([-30, -15, 15, 30, 45])
            if d > 0 and d not in opts:
                opts.append(d)
        rng.shuffle(opts)
        ans_idx = opts.index(effort)
        answer = chr(ord("A") + ans_idx)
        uw, ul = "lb", "ft"
        q = rng.choice(_INCLINED_TEMPLATES).format(
            w=W, h=h, L=L, uw=uw, ul=ul,
            a=opts[0], b=opts[1], c=opts[2], d=opts[3])
        if cfg["give_hint"]:
            q = q + _HINT_INCLINED
        img = self._render_inclined(W, h, L, rng)
        return q, answer, img

    def _gen_lever_dist(self, rng, cfg, level):
        # M1 * d1 = M2 * d2; d1 + d2 = dtot
        for _ in range(40):
            m1 = rng.choice([2, 3, 4, 5, 6, 8, 10])
            m2 = rng.choice([2, 3, 4, 5, 6, 8, 10])
            if m1 == m2:
                continue
            dtot_mul = rng.choice([3, 4, 5, 6])
            dtot = (m1 + m2) * dtot_mul // math.gcd(m1, m2) // (m1 + m2) * (m1 + m2)
            # Simpler: choose dtot as multiple of (m1+m2)
            dtot = (m1 + m2) * rng.choice([1, 2, 3])
            d1 = m2 * dtot / (m1 + m2)
            if float(d1).is_integer():
                d1 = int(d1)
                break
        else:
            m1, m2, dtot = 2, 3, 5
            d1 = 3
        # Distractors
        opts_set = {d1, dtot - d1, dtot // 2, m1, m2, dtot}
        opts_set.discard(d1)
        opts = [d1] + list(opts_set)[:3]
        while len(opts) < 4:
            d = d1 + rng.choice([-2, -1, 1, 2])
            if d > 0 and d != d1 and d not in opts:
                opts.append(d)
        rng.shuffle(opts)
        ans_idx = opts.index(d1)
        answer = chr(ord("A") + ans_idx)
        uw, ul = "lb", "cm"
        # We have a typo in the templates ({b) should be {b}); rebuild text inline for safety
        q = (
            f"A lever balances at the fulcrum. Mass M1 = {m1} {uw} is on the "
            f"left and mass M2 = {m2} {uw} is on the right. The total distance "
            f"between the two masses is {dtot} {ul}. At what distance from the "
            f"left mass should the fulcrum be placed to balance the lever? "
            f"(A) {opts[0]} {ul} (B) {opts[1]} {ul} (C) {opts[2]} {ul} "
            f"(D) {opts[3]} {ul}. Reply with the letter (A/B/C/D) in <answer>...</answer>."
        )
        if cfg["give_hint"]:
            q = q + _HINT_LEVER
        img = self._render_lever_dist(m1, m2, d1, dtot, uw, ul, rng)
        return q, answer, img

    # =========================================================== #
    # Renderers
    # =========================================================== #

    def _render_gears(self, n, driver_dir, marked_idx, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(2 + 1.6 * n, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Gears arranged in a horizontal chain
        radius = 0.6
        cx = 0
        centers = []
        for i in range(n):
            cx_i = i * 2 * radius * 0.95  # touching
            centers.append((cx_i, 0))
        # Draw each gear as a serrated circle
        for i, (gx, gy) in enumerate(centers):
            color = "#ff8c00" if i == marked_idx else "#bdbdbd"
            edge = "#212121"
            # Body
            circ = mpatches.Circle((gx, gy), radius * 0.85,
                                    facecolor=color, edgecolor=edge, linewidth=1.4, zorder=2)
            ax.add_patch(circ)
            # Teeth: short rectangles around the circumference
            n_teeth = 12
            for k in range(n_teeth):
                a = 2 * math.pi * k / n_teeth
                rx = math.cos(a) * radius * 0.95
                ry = math.sin(a) * radius * 0.95
                tw = 0.10
                # Use a small triangle tooth
                tx1 = gx + math.cos(a) * radius * 0.85
                ty1 = gy + math.sin(a) * radius * 0.85
                tx2 = gx + math.cos(a + 0.13) * radius * 1.02
                ty2 = gy + math.sin(a + 0.13) * radius * 1.02
                tx3 = gx + math.cos(a - 0.13) * radius * 1.02
                ty3 = gy + math.sin(a - 0.13) * radius * 1.02
                tooth = mpatches.Polygon([(tx1, ty1), (tx2, ty2), (tx3, ty3)],
                                          closed=True, facecolor=color,
                                          edgecolor=edge, linewidth=0.8, zorder=2)
                ax.add_patch(tooth)
            # Center axle
            ax.plot([gx], [gy], "o", color=edge, markersize=4, zorder=4)
        # Driver gear arrow (showing rotation direction)
        driver_x, driver_y = centers[0]
        # Draw a short curved arrow indicating CW or CCW
        arc_radius = radius * 0.5
        if driver_dir == +1:  # CW
            theta1, theta2 = 90, 350
            arrow_dx, arrow_dy = 0.05, -0.10
        else:  # CCW
            theta1, theta2 = 350, 90
            arrow_dx, arrow_dy = -0.05, -0.10
        from matplotlib.patches import Arc, FancyArrowPatch
        arc = Arc((driver_x, driver_y), 2 * arc_radius, 2 * arc_radius,
                  angle=0, theta1=theta1, theta2=theta2, color="#1976d2", lw=2)
        ax.add_patch(arc)
        # Arrow tip
        if driver_dir == +1:  # CW: tip at angle ~ 350°
            tip_a = math.radians(350)
            base_a = math.radians(330)
        else:
            tip_a = math.radians(90 + 30)
            base_a = math.radians(90 + 50)
        tip = (driver_x + arc_radius * math.cos(tip_a),
               driver_y + arc_radius * math.sin(tip_a))
        base = (driver_x + arc_radius * math.cos(base_a),
                driver_y + arc_radius * math.sin(base_a))
        ax.annotate("", xy=tip, xytext=base,
                    arrowprops=dict(arrowstyle="->", color="#1976d2", lw=2))
        # Label gears
        for i, (gx, gy) in enumerate(centers):
            label = "Driver" if i == 0 else (f"G{i+1}" if i != marked_idx else "Marked")
            ax.text(gx, gy - radius * 1.4, label,
                    fontsize=9, ha="center", va="top", color="#212121")
        # Bounds
        ax.set_xlim(-radius * 1.5, centers[-1][0] + radius * 1.5)
        ax.set_ylim(-radius * 2.0, radius * 2.0)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_balance(self, K, dK, unknown, dU, u, rng) -> Image.Image:
        # Lever sits on a triangular fulcrum. Known weight on left, unknown on right.
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Lever
        max_d = max(dK, dU)
        ax.plot([-max_d - 0.5, max_d + 0.5], [0, 0],
                color="#212121", linewidth=3)
        # Fulcrum (triangle below 0)
        tri = mpatches.Polygon(
            [(0, 0), (-0.4, -0.7), (0.4, -0.7)], closed=True,
            facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0)
        ax.add_patch(tri)
        # Hatching below fulcrum
        for k in range(6):
            xa = -0.4 + k * 0.16
            ax.plot([xa, xa - 0.08], [-0.7, -0.85],
                    color="#212121", linewidth=0.8)
        ax.plot([-0.5, 0.5], [-0.85, -0.85], color="#212121", linewidth=1)
        # Known weight on left
        kbox = mpatches.Rectangle((-dK - 0.25, 0.0), 0.5, 0.5,
                                   facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(kbox)
        ax.text(-dK, 0.25, f"{K} {u}", fontsize=10, ha="center", va="center",
                fontweight="bold")
        # Unknown weight on right
        ubox = mpatches.Rectangle((dU - 0.25, 0.0), 0.5, 0.5,
                                   facecolor="#fff59d", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(ubox)
        ax.text(dU, 0.25, "?", fontsize=14, ha="center", va="center",
                fontweight="bold")
        # Distance markings
        ax.annotate("", xy=(-dK, -0.05), xytext=(0, -0.05),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
        ax.text(-dK / 2, -0.18, f"{dK}", fontsize=9, ha="center", va="top",
                color="#1976d2")
        ax.annotate("", xy=(dU, -0.05), xytext=(0, -0.05),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
        ax.text(dU / 2, -0.18, f"{dU}", fontsize=9, ha="center", va="top",
                color="#1976d2")
        ax.set_xlim(-max_d - 1, max_d + 1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_sleds(self, widths_for_letter, rng) -> Image.Image:
        n = len(widths_for_letter)
        fig, ax = plt.subplots(figsize=(2 + 1.6 * n, 3.2), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        spacing = 1.6
        x0 = 0
        for i, w in enumerate(widths_for_letter):
            cx = x0 + i * spacing
            # Sled body
            body = mpatches.Rectangle((cx - 0.5, 0.4), 1.0, 0.3,
                                       facecolor="#a1887f", edgecolor="#212121", linewidth=1.0)
            ax.add_patch(body)
            # Two runners (drawn as vertical-thickness rectangles)
            half_w = w / 2.0 * 0.30
            runner1 = mpatches.Rectangle((cx - 0.4, 0.1), half_w * 2, 0.2,
                                         facecolor="#37474f", edgecolor="#212121", linewidth=1.0)
            runner2 = mpatches.Rectangle((cx + 0.4 - half_w * 2, 0.1), half_w * 2, 0.2,
                                         facecolor="#37474f", edgecolor="#212121", linewidth=1.0)
            ax.add_patch(runner1)
            ax.add_patch(runner2)
            ax.text(cx, -0.05, chr(ord("A") + i),
                    fontsize=14, fontweight="bold", ha="center", va="top")
        # Ground (icy slope hint)
        ax.plot([-0.8, x0 + (n - 1) * spacing + 0.8], [0.05, 0.05],
                color="#212121", linewidth=1.0)
        ax.set_xlim(-1.0, x0 + (n - 1) * spacing + 1.0)
        ax.set_ylim(-0.4, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_pulley(self, movable_letter, rng) -> Image.Image:
        # Two arrangements side by side: A left, B right.
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Support bar
        ax.plot([-3, 3], [3, 3], color="#212121", linewidth=3)
        # Hatching
        for k in range(20):
            xa = -3 + k * 0.3
            ax.plot([xa, xa - 0.15], [3, 3.3], color="#212121", linewidth=0.7)
        # Arrangement A: left
        # Arrangement B: right
        for letter, cx in [("A", -1.5), ("B", 1.5)]:
            if letter == movable_letter:
                # Movable pulley: rope goes over a fixed support pulley, then
                # down around a movable pulley, weight hangs from movable axle.
                # Top fixed pulley at (cx, 2.6)
                tpulley = mpatches.Circle((cx, 2.6), 0.18,
                                          facecolor="#90caf9", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(tpulley)
                # Movable pulley at (cx, 1.0)
                mpulley = mpatches.Circle((cx, 1.0), 0.18,
                                          facecolor="#90caf9", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(mpulley)
                # Rope
                ax.plot([cx - 0.5, cx - 0.18], [2.6, 2.6], color="#212121", linewidth=1.2)
                ax.plot([cx - 0.5, cx - 0.5], [2.6, 0.4], color="#212121", linewidth=1.2)
                ax.plot([cx - 0.5, cx - 0.18], [1.0 - 0.0, 1.0], color="#212121", linewidth=1.2)
                ax.plot([cx + 0.18, cx + 0.18], [1.0, 2.6], color="#212121", linewidth=1.2)
                ax.plot([cx + 0.18, cx + 0.5], [2.6, 2.6], color="#212121", linewidth=1.2)
                # Weight at bottom of movable pulley
                weight = mpatches.Rectangle((cx - 0.3, 0.4), 0.6, 0.4,
                                            facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(weight)
                ax.text(cx, 0.6, "W", fontsize=11, ha="center", va="center",
                        fontweight="bold")
                # Effort arrow at left end of rope
                ax.annotate("", xy=(cx - 0.5, 0.0), xytext=(cx - 0.5, 0.5),
                            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
                ax.text(cx - 0.5, -0.15, "Effort", fontsize=8, ha="center",
                        va="top", color="#c0392b")
            else:
                # Single fixed pulley: rope over pulley, weight on one end, effort on other
                tpulley = mpatches.Circle((cx, 2.6), 0.18,
                                          facecolor="#90caf9", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(tpulley)
                # Rope
                ax.plot([cx - 0.18, cx - 0.18], [0.6, 2.6],
                        color="#212121", linewidth=1.2)
                ax.plot([cx + 0.18, cx + 0.18], [0.6, 2.6],
                        color="#212121", linewidth=1.2)
                # Weight on right
                weight = mpatches.Rectangle((cx + 0.18 - 0.3, 0.2), 0.6, 0.4,
                                            facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(weight)
                ax.text(cx + 0.18, 0.4, "W", fontsize=11, ha="center", va="center",
                        fontweight="bold")
                # Effort arrow on left
                ax.annotate("", xy=(cx - 0.18, 0.0), xytext=(cx - 0.18, 0.5),
                            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
                ax.text(cx - 0.18, -0.15, "Effort", fontsize=8, ha="center",
                        va="top", color="#c0392b")
            ax.text(cx, 3.5, letter, fontsize=18, fontweight="bold",
                    ha="center", va="center")
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-0.6, 4.0)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_circuit(self, kind, r1, r2, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Battery on the left
        ax.plot([0, 0], [-0.2, 0.2], color="#212121", linewidth=3)
        ax.plot([0.1, 0.1], [-0.4, 0.4], color="#212121", linewidth=2)
        ax.text(-0.25, 0, "Battery", fontsize=8, ha="right", va="center")
        if kind == "series":
            # series loop: battery -> R1 -> R2 -> back
            # Top wire
            ax.plot([0.1, 5], [0.4, 0.4], color="#212121", linewidth=1.5)
            ax.plot([5, 5], [0.4, -0.4], color="#212121", linewidth=1.5)
            ax.plot([0.1, 5], [-0.4, -0.4], color="#212121", linewidth=1.5)
            # R1 box
            r1_box = mpatches.Rectangle((1.5, 0.32), 1.0, 0.16,
                                         facecolor="#fff59d", edgecolor="#212121", linewidth=1.2)
            ax.add_patch(r1_box)
            ax.text(2.0, 0.65, f"R1 = {r1} Ω", fontsize=10, ha="center", va="center")
            # R2 box
            r2_box = mpatches.Rectangle((3.0, 0.32), 1.0, 0.16,
                                         facecolor="#fff59d", edgecolor="#212121", linewidth=1.2)
            ax.add_patch(r2_box)
            ax.text(3.5, 0.65, f"R2 = {r2} Ω", fontsize=10, ha="center", va="center")
            ax.set_xlim(-1.0, 5.5)
            ax.set_ylim(-1.0, 1.2)
        else:  # parallel
            # main loop
            ax.plot([0.1, 5], [0.4, 0.4], color="#212121", linewidth=1.5)
            ax.plot([5, 5], [0.4, -0.4], color="#212121", linewidth=1.5)
            ax.plot([0.1, 5], [-0.4, -0.4], color="#212121", linewidth=1.5)
            # Two parallel branches between (2, 0.4) and (3.5, 0.4) and (2, -0.4) and (3.5, -0.4)
            ax.plot([2, 2], [0.4, -0.4], color="#212121", linewidth=1.5)
            ax.plot([3.5, 3.5], [0.4, -0.4], color="#212121", linewidth=1.5)
            # R1 inside top branch
            r1_box = mpatches.Rectangle((2.4, 0.05), 0.6, 0.3,
                                         facecolor="#fff59d", edgecolor="#212121", linewidth=1.2)
            ax.add_patch(r1_box)
            ax.text(2.7, 0.20, f"R1\n{r1} Ω", fontsize=9, ha="center", va="center")
            # R2 inside bottom branch
            r2_box = mpatches.Rectangle((2.4, -0.35), 0.6, 0.3,
                                         facecolor="#fff59d", edgecolor="#212121", linewidth=1.2)
            ax.add_patch(r2_box)
            ax.text(2.7, -0.20, f"R2\n{r2} Ω", fontsize=9, ha="center", va="center")
            ax.set_xlim(-1.0, 5.5)
            ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_spring(self, W1, x1, ux, uw, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(4, 5.2), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Top mounting bar
        ax.plot([-1, 1], [3, 3], color="#212121", linewidth=3)
        for k in range(10):
            xa = -1 + k * 0.2
            ax.plot([xa, xa - 0.1], [3, 3.2], color="#212121", linewidth=0.7)
        # Spring (zig-zag)
        n_zigs = 8
        spring_top = 3.0
        spring_bot = 1.2
        x_amp = 0.35
        ys = np.linspace(spring_top, spring_bot, n_zigs * 2 + 1)
        xs = np.array([x_amp if i % 2 == 0 else -x_amp for i in range(len(ys))])
        ax.plot(xs, ys, color="#212121", linewidth=1.5)
        # Weight at bottom
        weight = mpatches.Rectangle((-0.5, 0.6), 1.0, 0.6,
                                    facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(weight)
        ax.text(0, 0.9, f"{W1} {uw}", fontsize=12, ha="center", va="center",
                fontweight="bold")
        # Annotate stretch
        ax.annotate("", xy=(-1.5, spring_bot), xytext=(-1.5, spring_top),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.2))
        ax.text(-1.7, (spring_top + spring_bot) / 2, f"stretch = {x1} {ux}",
                fontsize=10, ha="right", va="center", color="#1976d2")
        ax.set_xlim(-2.5, 1.5)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_piston(self, n_compress, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Two sub-figures: position A (left), position B (right).
        # Cylinder for A
        cylA_l, cylA_r = -3, -1
        ax.plot([cylA_l, cylA_l], [0, 1], color="#212121", linewidth=2)
        ax.plot([cylA_l, cylA_r + 0.4], [0, 0], color="#212121", linewidth=2)
        ax.plot([cylA_l, cylA_r + 0.4], [1, 1], color="#212121", linewidth=2)
        # Piston A (at right end - leftmost compression)
        pis_x = cylA_r
        ax.plot([pis_x, pis_x], [0, 1], color="#212121", linewidth=4)
        # Gas region in A
        gas = mpatches.Rectangle((cylA_l, 0), pis_x - cylA_l, 1,
                                  facecolor="#bbdefb", alpha=0.6, edgecolor="none")
        ax.add_patch(gas)
        ax.text((cylA_l + pis_x) / 2, 0.5, "Gas", fontsize=10, ha="center", va="center")
        ax.text((cylA_l + pis_x) / 2, -0.3, "Position A", fontsize=11,
                ha="center", va="top", fontweight="bold")
        # Cylinder for B
        cylB_l, cylB_r = 1, 3
        ax.plot([cylB_l, cylB_l], [0, 1], color="#212121", linewidth=2)
        ax.plot([cylB_l, cylB_r + 0.4], [0, 0], color="#212121", linewidth=2)
        ax.plot([cylB_l, cylB_r + 0.4], [1, 1], color="#212121", linewidth=2)
        # Piston B compresses to 1/n volume; original length = (cylB_r - cylB_l) = 2
        new_len = 2 / n_compress
        pisB_x = cylB_l + new_len
        ax.plot([pisB_x, pisB_x], [0, 1], color="#212121", linewidth=4)
        gasB = mpatches.Rectangle((cylB_l, 0), pisB_x - cylB_l, 1,
                                   facecolor="#bbdefb", alpha=0.6, edgecolor="none")
        ax.add_patch(gasB)
        ax.text((cylB_l + pisB_x) / 2, 0.5, "Gas", fontsize=9, ha="center", va="center")
        ax.text((cylB_l + cylB_r) / 2, -0.3, "Position B", fontsize=11,
                ha="center", va="top", fontweight="bold")
        # Arrow showing piston motion
        ax.annotate("", xy=(pisB_x, 1.3), xytext=(cylB_r, 1.3),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.set_xlim(-3.5, 4)
        ax.set_ylim(-0.8, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_inclined(self, W, h, L, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Right-triangle ramp: base = sqrt(L^2 - h^2), height = h
        # Use scaled coordinates so the ramp fits the figure
        scale = 3.0 / L  # fit total ramp length in 3 units
        sh = h * scale
        sb = math.sqrt(max(0, L * L - h * h)) * scale
        # Triangle: (0,0), (sb, 0), (0, sh)? Or hypotenuse from (0, sh) to (sb, 0)?
        # Standard: ramp goes from ground to platform. Base on x-axis from 0 to sb, vertical leg from (0, sh) to (sb, 0)? No — ramp = hypotenuse from (0, 0) to (sb, sh)? Easier:
        # Place platform at top-left and ground line at bottom; hypotenuse from (sb, 0) to (0, sh) — this is the ramp surface.
        ramp = mpatches.Polygon([(0, 0), (sb, 0), (0, sh)], closed=True,
                                facecolor="#d7ccc8", edgecolor="#212121", linewidth=1.4)
        ax.add_patch(ramp)
        # Label height h (vertical leg)
        ax.annotate("", xy=(-0.2, 0), xytext=(-0.2, sh),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
        ax.text(-0.4, sh / 2, f"h = {h} ft", fontsize=10, ha="right", va="center",
                color="#1976d2")
        # Label ramp length on hypotenuse
        mid_x = sb / 2
        mid_y = sh / 2
        ax.text(mid_x + 0.2, mid_y + 0.2, f"L = {L} ft", fontsize=10,
                color="#1976d2")
        # Ball at the bottom of the ramp
        ball = mpatches.Circle((sb - 0.25, 0.2), 0.18,
                                facecolor="#bbdefb", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(ball)
        ax.text(sb - 0.25, -0.25, f"{W} lb",
                fontsize=10, ha="center", va="top", fontweight="bold")
        # Force arrow up the ramp
        # Direction of ramp from (sb, 0) toward (0, sh)
        dx, dy = -sb, sh
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        ax.annotate("", xy=(sb - 0.25 + ux * 0.7, 0.2 + uy * 0.7),
                    xytext=(sb - 0.25 + ux * 0.1, 0.2 + uy * 0.1),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.set_xlim(-1.2, sb + 0.5)
        ax.set_ylim(-0.6, sh + 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    def _render_lever_dist(self, m1, m2, d1, dtot, uw, ul, rng) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Lever from x=0 to x=dtot, with masses at endpoints and fulcrum at d1.
        # Scale to fit ~5 units of x.
        scale = 5.0 / max(dtot, 1)
        # Lever
        ax.plot([0, dtot * scale], [0, 0], color="#212121", linewidth=3)
        # Mass M1 on left
        m1box = mpatches.Rectangle((-0.25, 0), 0.5, 0.5,
                                    facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(m1box)
        ax.text(0, 0.25, f"{m1} {uw}", fontsize=10, ha="center", va="center",
                fontweight="bold")
        # Mass M2 on right
        m2box = mpatches.Rectangle((dtot * scale - 0.25, 0), 0.5, 0.5,
                                    facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(m2box)
        ax.text(dtot * scale, 0.25, f"{m2} {uw}", fontsize=10, ha="center", va="center",
                fontweight="bold")
        # Fulcrum at d1 from left
        fx = d1 * scale
        tri = mpatches.Polygon(
            [(fx, 0), (fx - 0.4, -0.7), (fx + 0.4, -0.7)], closed=True,
            facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0)
        ax.add_patch(tri)
        # Hatching
        for k in range(6):
            xa = fx - 0.4 + k * 0.16
            ax.plot([xa, xa - 0.08], [-0.7, -0.85], color="#212121", linewidth=0.8)
        ax.plot([fx - 0.5, fx + 0.5], [-0.85, -0.85], color="#212121", linewidth=1)
        # Distance label dtot
        ax.annotate("", xy=(0, -1.2), xytext=(dtot * scale, -1.2),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
        ax.text(dtot * scale / 2, -1.35, f"distance = {dtot} {ul}",
                fontsize=10, ha="center", va="top", color="#1976d2")
        ax.set_xlim(-1, dtot * scale + 1)
        ax.set_ylim(-1.7, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)

    # =========================================================== #
    # Shared helper: render 3 candidate pictograms in a row (A/B/C).
    # =========================================================== #
    def _render_3_candidates(self, draw_fns, labels=("A", "B", "C"), title=None):
        """draw_fns: list of N callables each receiving (ax, rng) and drawing
        a pictogram into the ax. labels: text labels under each pictogram.
        Returns a PIL.Image."""
        n = len(draw_fns)
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.5), dpi=120)
        if n == 1:
            axes = [axes]
        fig.patch.set_facecolor("#ffffff")
        for i, ax in enumerate(axes):
            ax.set_facecolor("#ffffff")
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_aspect("equal")
            draw_fns[i](ax)
            ax.text(0, -0.95, labels[i], fontsize=14, ha="center", va="top",
                    fontweight="bold")
            ax.axis("off")
        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        return self.fig_to_pil(fig, dpi=120)

    # =========================================================== #
    # NEW common-sense mechanical sub-templates
    # =========================================================== #

    def _gen_bird_drag(self, rng, cfg, level):
        # Three birds: tucked / spread / partially-spread wings.
        # Correct = tucked wings = least drag. Letters shuffled.
        wing_states = ["tucked", "spread", "partial"]
        rng.shuffle(wing_states)
        correct_idx = wing_states.index("tucked")
        answer = chr(ord("A") + correct_idx)

        def make_bird(state):
            def _draw(ax):
                # Body — ellipse
                body = mpatches.Ellipse((0, 0), 0.6, 0.3,
                                        facecolor="#a1887f", edgecolor="#212121", linewidth=1.2)
                ax.add_patch(body)
                # Beak (small triangle)
                ax.add_patch(mpatches.Polygon([(0.30, 0.05), (0.45, 0), (0.30, -0.05)],
                                              closed=True, facecolor="#ffb74d",
                                              edgecolor="#212121", linewidth=0.8))
                # Eye
                ax.plot([0.20], [0.05], "o", color="#212121", markersize=3)
                # Wings
                if state == "tucked":
                    # Wings folded against body — small triangle
                    w1 = mpatches.Polygon([(-0.10, 0.10), (-0.30, 0.20), (-0.20, 0.05)],
                                          closed=True, facecolor="#8d6e63",
                                          edgecolor="#212121", linewidth=1.0)
                    ax.add_patch(w1)
                elif state == "spread":
                    # Wings extended outward
                    w1 = mpatches.Polygon([(-0.10, 0.10), (-0.55, 0.55), (-0.30, 0.10)],
                                          closed=True, facecolor="#8d6e63",
                                          edgecolor="#212121", linewidth=1.0)
                    w2 = mpatches.Polygon([(-0.10, 0.10), (0.20, 0.55), (-0.05, 0.10)],
                                          closed=True, facecolor="#8d6e63",
                                          edgecolor="#212121", linewidth=1.0)
                    ax.add_patch(w1); ax.add_patch(w2)
                else:
                    # partial spread
                    w1 = mpatches.Polygon([(-0.10, 0.10), (-0.40, 0.30), (-0.25, 0.05)],
                                          closed=True, facecolor="#8d6e63",
                                          edgecolor="#212121", linewidth=1.0)
                    w2 = mpatches.Polygon([(-0.10, 0.10), (0.15, 0.30), (-0.05, 0.10)],
                                          closed=True, facecolor="#8d6e63",
                                          edgecolor="#212121", linewidth=1.0)
                    ax.add_patch(w1); ax.add_patch(w2)
                # Tail
                tail = mpatches.Polygon([(-0.30, 0), (-0.55, 0.10), (-0.55, -0.10)],
                                        closed=True, facecolor="#a1887f",
                                        edgecolor="#212121", linewidth=1.0)
                ax.add_patch(tail)
            return _draw

        draws = [make_bird(s) for s in wing_states]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three birds are flying at the same speed in the same direction. "
             "Each bird has its wings in a different position. Which bird "
             "experiences the least air resistance? "
             "(A) Bird A (B) Bird B (C) Bird C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: less wing surface in the airflow = less drag."
        return q, answer, img

    def _gen_drone_stability(self, rng, cfg, level):
        # Three drones, identical except for cable length to attached object.
        # Correct = shortest cable = most stable.
        lens = [rng.uniform(0.3, 0.45), rng.uniform(0.55, 0.70), rng.uniform(0.80, 0.95)]
        rng.shuffle(lens)
        shortest_idx = lens.index(min(lens))
        answer = chr(ord("A") + shortest_idx)

        def make_drone(cable_len):
            def _draw(ax):
                # Drone body (rectangle with two top rotors)
                body = mpatches.Rectangle((-0.30, 0.55), 0.60, 0.10,
                                          facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(body)
                # Rotors
                for rx in (-0.30, 0.30):
                    ax.plot([rx - 0.18, rx + 0.18], [0.70, 0.70],
                            color="#212121", linewidth=2)
                # Cable
                ax.plot([0, 0], [0.55, 0.55 - cable_len],
                        color="#37474f", linewidth=1.5)
                # Object hanging
                obj = mpatches.Rectangle((-0.18, 0.55 - cable_len - 0.20),
                                         0.36, 0.20,
                                         facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(obj)
            return _draw

        draws = [make_drone(l) for l in lens]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three drones, each with an identical attached object, are hovering. "
             "They differ only in the length of cable between drone and object. "
             "Which drone is the most stable? "
             "(A) Drone A (B) Drone B (C) Drone C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: shorter cable → load is closer to the drone's center of gravity → more stable."
        return q, answer, img

    def _gen_parcel_pivot(self, rng, cfg, level):
        # Three parcels suspended by ropes, each rope-attachment offset
        # differently from the parcel center.
        # Correct = most stable = rope attached at the very edge / highest point
        # (per reference qid 428: the parcel with the highest pivot point above
        # its center of mass is the most stable).
        offsets = [
            (0.0, "center"),     # rope attached at center top
            (0.40, "edge"),      # rope attached at edge — *unstable*
            (-0.40, "edge2"),    # rope attached at other edge
        ]
        rng.shuffle(offsets)
        # The parcel where the pivot is directly above the center of mass is
        # most stable (offset = 0).
        center_idx = next(i for i, (o, _) in enumerate(offsets) if abs(o) < 1e-6)
        answer = chr(ord("A") + center_idx)

        def make_parcel(off):
            def _draw(ax):
                # Ceiling at top
                ax.plot([-0.7, 0.7], [0.85, 0.85], color="#212121", linewidth=2)
                # Hatching
                for k in range(8):
                    xa = -0.7 + k * 0.18
                    ax.plot([xa, xa - 0.08], [0.85, 0.95],
                            color="#212121", linewidth=0.6)
                # Rope (from ceiling down to top of parcel)
                ax.plot([off, off], [0.85, 0.10], color="#212121", linewidth=1.4)
                # Parcel — rectangle centered on x=0
                par = mpatches.Rectangle((-0.40, -0.30), 0.80, 0.40,
                                         facecolor="#a1887f", edgecolor="#212121", linewidth=1.4)
                ax.add_patch(par)
                # Knot at attachment
                ax.plot([off], [0.10], "o", color="#212121", markersize=4)
            return _draw

        draws = [make_parcel(o) for o, _ in offsets]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three parcels are suspended by a rope from the ceiling. The rope "
             "attaches to a different point on the top of each parcel. Which "
             "parcel is the most stable? "
             "(A) A (B) B (C) C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: the most stable parcel hangs with its center of mass directly below the attachment point."
        return q, answer, img

    def _gen_halligan_lever(self, rng, cfg, level):
        # Three halligan-style pry bars; vary the distance from the handle (force)
        # to the prying tip (load). Correct = longest bar = least effort.
        lens = [0.5, 0.75, 1.0]
        rng.shuffle(lens)
        longest_idx = lens.index(max(lens))
        answer = chr(ord("A") + longest_idx)

        def make_halligan(L):
            def _draw(ax):
                # Bar from (-L/2, 0) to (+L/2, 0), with prying claw at right end
                ax.plot([-L / 2, L / 2], [0.10, 0.10],
                        color="#37474f", linewidth=4)
                # Handle grip (left end, small box)
                grip = mpatches.Rectangle((-L / 2 - 0.06, -0.04), 0.10, 0.28,
                                          facecolor="#5d4037", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(grip)
                # Claw at right end (V shape)
                ax.add_patch(mpatches.Polygon([(L / 2, 0.0), (L / 2 + 0.12, 0.20),
                                              (L / 2 + 0.05, 0.0)],
                                              closed=True, facecolor="#37474f",
                                              edgecolor="#212121", linewidth=1.0))
                ax.add_patch(mpatches.Polygon([(L / 2, 0.20), (L / 2 + 0.12, 0.20),
                                              (L / 2 + 0.05, 0.10)],
                                              closed=True, facecolor="#37474f",
                                              edgecolor="#212121", linewidth=1.0))
                # Force arrow on grip
                ax.annotate("", xy=(-L / 2 - 0.03, -0.20), xytext=(-L / 2 - 0.03, -0.05),
                            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
            return _draw

        draws = [make_halligan(l) for l in lens]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three halligan pry tools have different bar lengths. Which one "
             "would require the least effort to pry open the same door? "
             "(A) Tool A (B) Tool B (C) Tool C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: a longer lever arm gives more mechanical advantage."
        return q, answer, img

    def _gen_spade_carry(self, rng, cfg, level):
        # Three workers carrying a spade at different distances from their body.
        # Correct = spade closest to body = least effort.
        offsets = [0.05, 0.30, 0.55]
        rng.shuffle(offsets)
        closest_idx = offsets.index(min(offsets))
        answer = chr(ord("A") + closest_idx)

        def make_carry(off):
            def _draw(ax):
                # Stick figure: head + body + arm
                # Head
                ax.add_patch(mpatches.Circle((0, 0.55), 0.12,
                                             facecolor="#ffe0b2", edgecolor="#212121", linewidth=1.0))
                # Body
                ax.plot([0, 0], [0.43, -0.10], color="#212121", linewidth=2)
                # Legs
                ax.plot([0, -0.12], [-0.10, -0.50], color="#212121", linewidth=2)
                ax.plot([0, 0.12], [-0.10, -0.50], color="#212121", linewidth=2)
                # Arm extends to the right horizontally to grip the spade
                ax.plot([0, off], [0.30, 0.30], color="#212121", linewidth=2)
                # Spade: long shaft + blade at top
                shaft_x = off
                ax.plot([shaft_x, shaft_x], [0.30, 0.65],
                        color="#5d4037", linewidth=3)
                # Blade (rectangle at top)
                blade = mpatches.Rectangle((shaft_x - 0.08, 0.65), 0.16, 0.18,
                                           facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0)
                ax.add_patch(blade)
                # Distance label
                ax.annotate("", xy=(off, 0.20), xytext=(0, 0.20),
                            arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
            return _draw

        draws = [make_carry(o) for o in offsets]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three workers carry the same spade in three different positions. "
             "Which method requires the least effort? "
             "(A) Method A (B) Method B (C) Method C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: carrying the load closer to the body reduces the torque on your arm."
        return q, answer, img

    def _gen_balloon_pressure(self, rng, cfg, level):
        # Two balloons: A fully inflated, B half inflated. (3-option MCQ A/B/C)
        # Correct = A (fully inflated has more pressure inside).
        # Randomize labels.
        states = ["full", "half"]
        rng.shuffle(states)
        full_idx = states.index("full")
        answer = chr(ord("A") + full_idx)

        def make_balloon(state):
            def _draw(ax):
                # Balloon (circle), tied off at bottom with small triangle
                if state == "full":
                    r = 0.55
                    fc = "#ef9a9a"
                else:
                    r = 0.32
                    fc = "#ef9a9a"
                ax.add_patch(mpatches.Circle((0, 0.10), r,
                                             facecolor=fc, edgecolor="#212121", linewidth=1.4))
                # tie
                ax.add_patch(mpatches.Polygon([(0, 0.10 - r), (-0.05, 0.10 - r - 0.10),
                                              (0.05, 0.10 - r - 0.10)],
                                              closed=True, facecolor="#ef9a9a",
                                              edgecolor="#212121", linewidth=1.0))
                # string
                ax.plot([0, 0], [0.10 - r - 0.10, -0.7],
                        color="#212121", linewidth=0.8)
            return _draw

        draws = [make_balloon(s) for s in states]
        img = self._render_3_candidates(draws[:2], labels=("A", "B"))
        # Note: only 2 balloons rendered; option C = Equal pressure.
        q = ("Two balloons. Balloon A is fully inflated and balloon B is half "
             "inflated. Which has more air pressure inside? "
             "(A) Balloon A (B) Balloon B (C) Equal. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: more air packed into the same elastic envelope = higher internal pressure."
        return q, answer, img

    def _gen_granary_volume(self, rng, cfg, level):
        # Three cylindrical granaries with different (radius, height). Correct
        # holds the most wheat (highest pi*r^2*h).
        opts = []
        for _ in range(3):
            r = rng.choice([0.20, 0.25, 0.30, 0.35, 0.40])
            h = rng.choice([0.40, 0.50, 0.60, 0.70, 0.80])
            v = math.pi * r * r * h
            opts.append((r, h, v))
        # Make sure max is unique
        for _ in range(20):
            vs = [v for _, _, v in opts]
            if vs.count(max(vs)) == 1:
                break
            opts = []
            for _ in range(3):
                r = rng.choice([0.20, 0.25, 0.30, 0.35, 0.40])
                h = rng.choice([0.40, 0.50, 0.60, 0.70, 0.80])
                v = math.pi * r * r * h
                opts.append((r, h, v))
        max_idx = max(range(3), key=lambda i: opts[i][2])
        answer = chr(ord("A") + max_idx)

        def make_granary(r, h):
            def _draw(ax):
                # Cylinder side: rectangle + top ellipse
                rect = mpatches.Rectangle((-r, -h / 2), 2 * r, h,
                                          facecolor="#d7ccc8", edgecolor="#212121", linewidth=1.2)
                ax.add_patch(rect)
                # Top ellipse
                top = mpatches.Ellipse((0, h / 2), 2 * r, 0.10,
                                       facecolor="#a1887f", edgecolor="#212121", linewidth=1.2)
                ax.add_patch(top)
                # Bottom dashed
                bot = mpatches.Ellipse((0, -h / 2), 2 * r, 0.10,
                                       facecolor="none", edgecolor="#212121",
                                       linewidth=0.8, linestyle="--")
                ax.add_patch(bot)
            return _draw

        draws = [make_granary(r, h) for r, h, _ in opts]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three cylindrical granaries have different radii and heights. "
             "Which granary can hold the most wheat? "
             "(A) Granary A (B) Granary B (C) Granary C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: cylinder volume = π r² h."
        return q, answer, img

    def _gen_helicopter_tilt(self, rng, cfg, level):
        # Helicopter must tilt to fly forward. The rotor must tilt the same
        # direction as desired motion. From reference qid 432, ans: B = clockwise.
        # We render the helicopter with a "fly forward" arrow.
        direction = rng.choice(["forward", "backward"])
        # If forward → tilt nose down (clockwise from a side view).
        # If backward → tilt nose up (anticlockwise).
        if direction == "forward":
            correct_text = "Clockwise (nose down)"
        else:
            correct_text = "Anticlockwise (nose up)"
        opts = ["Anticlockwise (nose up)", "Clockwise (nose down)", "Either direction works"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index(correct_text))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Helicopter body (ellipse) — flat for now
        body = mpatches.Ellipse((0, 0), 1.6, 0.5,
                                facecolor="#ffe082", edgecolor="#212121", linewidth=1.4)
        ax.add_patch(body)
        # Tail
        ax.add_patch(mpatches.Polygon(
            [(-0.8, 0), (-1.6, 0.10), (-1.6, -0.10)],
            closed=True, facecolor="#ffe082",
            edgecolor="#212121", linewidth=1.0))
        # Tail rotor (small)
        ax.add_patch(mpatches.Circle((-1.7, 0), 0.15,
                                     facecolor="none", edgecolor="#212121", linewidth=1.0))
        # Top rotor (long horizontal line)
        ax.plot([-1.0, 1.0], [0.40, 0.40], color="#212121", linewidth=2)
        ax.plot([0, 0], [0.25, 0.40], color="#212121", linewidth=2)
        # Direction-of-motion arrow
        if direction == "forward":
            ax.annotate("", xy=(2.0, -0.6), xytext=(1.0, -0.6),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.5))
            ax.text(1.5, -0.85, "Wants to fly forward (right)",
                    fontsize=10, ha="center", va="top", color="#c0392b")
        else:
            ax.annotate("", xy=(-2.0, -0.6), xytext=(-1.0, -0.6),
                        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.5))
            ax.text(-1.5, -0.85, "Wants to fly backward (left)",
                    fontsize=10, ha="center", va="top", color="#c0392b")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)
        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(opts))
        q = ("Examine the helicopter shown in the image. To move in the "
             "indicated direction, the helicopter must tilt about its center. "
             "Which way must the helicopter tilt? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: the rotor pushes air opposite the direction of motion; tilt nose toward the direction of travel."
        return q, answer, img

    def _gen_candle_jar(self, rng, cfg, level):
        # Three candles under jars. (1) small sealed (2) big sealed (3) small with holes
        # Smallest sealed jar extinguishes first (least oxygen).
        opts = ["small_sealed", "big_sealed", "small_holes"]
        rng.shuffle(opts)
        small_sealed_idx = opts.index("small_sealed")
        answer = chr(ord("A") + small_sealed_idx)

        def make_candle_jar(kind):
            def _draw(ax):
                if kind == "small_sealed":
                    rx, ry = 0.30, 0.50
                    holes = False
                elif kind == "big_sealed":
                    rx, ry = 0.55, 0.70
                    holes = False
                else:
                    rx, ry = 0.30, 0.50
                    holes = True
                # Jar: rectangle + ellipse top
                ax.plot([-rx, -rx], [-0.30, ry - 0.30], color="#212121", linewidth=1.4)
                ax.plot([rx, rx], [-0.30, ry - 0.30], color="#212121", linewidth=1.4)
                # rounded top
                top = mpatches.Arc((0, ry - 0.30), 2 * rx, 0.20,
                                   theta1=0, theta2=180,
                                   color="#212121", linewidth=1.4)
                ax.add_patch(top)
                # Holes (drawn as gaps if holes==True)
                if holes:
                    ax.plot([-rx + 0.05, -rx + 0.05], [-0.18, -0.10],
                            color="#ffffff", linewidth=2.4)
                    ax.plot([rx - 0.05, rx - 0.05], [-0.18, -0.10],
                            color="#ffffff", linewidth=2.4)
                    ax.text(0, ry - 0.20, "holes", fontsize=8, ha="center")
                # Candle inside
                ax.add_patch(mpatches.Rectangle(
                    (-0.05, -0.30), 0.10, 0.40,
                    facecolor="#fdd835", edgecolor="#212121", linewidth=1.0))
                # Flame
                ax.add_patch(mpatches.Polygon(
                    [(0, 0.15), (-0.05, 0.10), (0.05, 0.10)],
                    closed=True, facecolor="#ff7043",
                    edgecolor="#212121", linewidth=0.8))
                # Ground
                ax.plot([-0.7, 0.7], [-0.30, -0.30], color="#212121", linewidth=1.0)
            return _draw

        draws = [make_candle_jar(k) for k in opts]
        img = self._render_3_candidates(draws, labels=("A", "B", "C"))
        q = ("Three identical candles are lit. Each is covered by a different "
             "jar (a small sealed jar, a large sealed jar, or a small jar with "
             "holes). Which candle will extinguish first? "
             "(A) Candle A (B) Candle B (C) Candle C. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: smaller sealed volume = less oxygen = candle goes out first."
        return q, answer, img

    def _gen_pole_pressure_compare(self, rng, cfg, level):
        # Rectangle resting on two poles at unequal distances from CG. The pole
        # closer to the rectangle's center of mass bears more weight.
        # Correct = the pole closer to CG.
        # Randomize which letter corresponds.
        layout = rng.choice(["close_left", "close_right"])
        if layout == "close_left":
            # CG is at x=0; pole 1 at x=-0.05, pole 2 at x=0.40
            answer = "A"
        else:
            answer = "B"
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Rectangle
        rect = mpatches.Rectangle((-0.6, 0.30), 1.2, 0.30,
                                  facecolor="#a1887f", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(rect)
        # CG marker
        ax.plot([0], [0.45], "x", color="#c0392b", markersize=12, mew=2)
        ax.text(0, 0.62, "CG", fontsize=9, ha="center", color="#c0392b")
        # Two poles (orange)
        if layout == "close_left":
            p1_x, p2_x = -0.05, 0.40
        else:
            p1_x, p2_x = -0.40, 0.05
        for x, label in [(p1_x, "1"), (p2_x, "2")]:
            ax.add_patch(mpatches.Rectangle((x - 0.04, -0.30), 0.08, 0.60,
                                            facecolor="#ff8c00", edgecolor="#212121", linewidth=1.0))
            ax.text(x, -0.40, f"Pole {label}", fontsize=10, ha="center", va="top")
        # Ground
        ax.plot([-1.0, 1.0], [-0.30, -0.30], color="#212121", linewidth=1)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.6, 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)
        q = ("A heavy rectangular block rests on two orange poles. The "
             "rectangle's center of mass (marked CG) is shown. On which pole "
             "is the weight pressing down harder? "
             "(A) Pole 1 (B) Pole 2 (C) Equal. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: torque balance — the pole closer to the center of mass carries more weight."
        return q, answer, img

    # =========================================================== #
    # NEW physical-intuition diagrams
    # =========================================================== #

    def _gen_torricelli_outflow(self, rng, cfg, level):
        # Bucket with leak. As water level falls, outflow speed decreases.
        # Always: answer = "decreases" → we'll pick label B usually but shuffle.
        opts = ["Increases", "Stays the same", "Decreases"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("Decreases"))

        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Bucket body (rectangle)
        ax.plot([-0.6, -0.6], [-0.5, 0.7], color="#212121", linewidth=2)
        ax.plot([0.6, 0.6], [-0.5, 0.7], color="#212121", linewidth=2)
        ax.plot([-0.6, 0.6], [-0.5, -0.5], color="#212121", linewidth=2)
        # Water (filled)
        water_h = 0.5
        water = mpatches.Rectangle((-0.59, -0.49), 1.18, water_h,
                                   facecolor="#90caf9", edgecolor="none", alpha=0.7)
        ax.add_patch(water)
        # Water surface line
        ax.plot([-0.6, 0.6], [-0.49 + water_h, -0.49 + water_h],
                color="#1976d2", linewidth=1.2)
        # Hole near the bottom on right side
        ax.plot([0.55, 0.65], [-0.30, -0.30], color="#ffffff", linewidth=4)
        # Outflow stream (parabolic)
        xs = np.linspace(0.65, 1.10, 20)
        ys = -0.30 - 0.5 * (xs - 0.65) ** 2 / 0.45
        ax.plot(xs, ys, color="#1976d2", linewidth=1.5)
        ax.text(0.9, -0.60, "outflow", fontsize=9, ha="center", color="#1976d2")
        # Annotation: water level decreasing arrow
        ax.annotate("", xy=(-0.85, -0.30), xytext=(-0.85, 0.10),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4))
        ax.text(-1.0, -0.10, "level\nfalling", fontsize=8, ha="right", va="center",
                color="#c0392b")
        ax.set_xlim(-1.4, 1.3)
        ax.set_ylim(-0.9, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(opts))
        q = ("A bucket of water has a small leak in its lower side. As the "
             "water level inside the bucket drops, what happens to the speed "
             "of water exiting the leak? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: the outflow speed depends on the water height above the hole (Torricelli's law)."
        return q, answer, img

    def _gen_magnifying_glass(self, rng, cfg, level):
        # Magnifying glass at distance D inches above wood; wood smokes at D.
        # To start a fire (focus the rays to a point) → bring lens to D inches
        # (= focal point). Distractors: shorter / longer.
        focal = rng.choice([10, 12, 15, 18, 20])
        opts_vals = [focal, focal - 5, focal + 10]
        rng.shuffle(opts_vals)
        ans_idx = opts_vals.index(focal)
        answer = chr(ord("A") + ans_idx)

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Sun (rays from top)
        ax.plot([-1.0, 1.0], [1.5, 1.5], color="#fdd835", linewidth=4)
        for x in np.linspace(-0.9, 0.9, 6):
            ax.plot([x, x], [1.5, 1.0], color="#fdd835", linewidth=1.0)
        # Magnifying glass: ellipse + handle
        lens = mpatches.Ellipse((0, 0.6), 0.8, 0.20,
                                facecolor="#bbdefb", edgecolor="#212121", linewidth=1.2, alpha=0.5)
        ax.add_patch(lens)
        ax.add_patch(mpatches.Rectangle((0.40, 0.55), 0.45, 0.10,
                                        facecolor="#5d4037", edgecolor="#212121", linewidth=1.0))
        # Light cone going down (converging)
        ax.add_patch(mpatches.Polygon(
            [(-0.40, 0.50), (0.40, 0.50), (0.10, -0.30), (-0.10, -0.30)],
            closed=True, facecolor="#ffe082", alpha=0.4, edgecolor="none"))
        # Wood / smoke
        ax.add_patch(mpatches.Rectangle((-0.5, -0.55), 1.0, 0.20,
                                        facecolor="#5d4037", edgecolor="#212121", linewidth=1.0))
        # Smoke wisps
        ax.text(0, -0.25, "smoke", fontsize=9, ha="center", color="#9e9e9e")
        # Distance label
        ax.annotate("", xy=(-0.85, -0.40), xytext=(-0.85, 0.55),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
        ax.text(-1.0, 0.05, f"{focal} in", fontsize=10, ha="right",
                va="center", color="#1976d2")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.8, 1.7)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {v} in"
                             for i, v in enumerate(opts_vals))
        q = (f"A camper holds a magnifying glass {focal} inches above a piece "
             f"of wood. At this distance, the wood begins to smoke. To start "
             f"a fire, how far should the lens be held above the wood? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: the wood smokes when the rays converge to the focal point — the same distance starts the fire."
        return q, answer, img

    def _gen_bridge_deflection(self, rng, cfg, level):
        # Simple-supported bridge — deflection peaks at the middle.
        opts = ["Under the supports", "At the middle of the bridge",
                "Below where the cars are parked", "All parts equally"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("At the middle of the bridge"))

        fig, ax = plt.subplots(figsize=(6.5, 3.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Two supports (towers) on left and right
        for sx in (-1.5, 1.5):
            ax.add_patch(mpatches.Polygon(
                [(sx - 0.18, -0.6), (sx + 0.18, -0.6), (sx, 0.0)],
                closed=True, facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
        # Bridge deck (curve sagging in middle)
        xs = np.linspace(-1.5, 1.5, 50)
        # parabola dipping at center
        ys = 0.05 - 0.10 * (1 - (xs / 1.5) ** 2)
        ax.plot(xs, ys, color="#212121", linewidth=3)
        # Cars on bridge — two identical at symmetric positions
        for cx in (-0.7, 0.7):
            cy = 0.05 - 0.10 * (1 - (cx / 1.5) ** 2) + 0.05
            ax.add_patch(mpatches.Rectangle((cx - 0.25, cy), 0.5, 0.20,
                                            facecolor="#bbdefb", edgecolor="#212121", linewidth=1.0))
            # Wheels
            for wx in (-0.18, 0.18):
                ax.add_patch(mpatches.Circle((cx + wx, cy), 0.05,
                                             facecolor="#212121", edgecolor="#212121"))
        # Ground
        ax.plot([-2.5, 2.5], [-0.6, -0.6], color="#212121", linewidth=1)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-0.9, 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(opts))
        q = ("Two identical cars are placed symmetrically on a long bridge "
             "with two supports. Where does the bridge experience the most "
             "deflection (downward sag)? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: a uniformly-loaded simply-supported beam sags most at its midpoint."
        return q, answer, img

    def _gen_convection_house(self, rng, cfg, level):
        # Cold air sinks (heavier). Correct: cold air enters along the floor.
        opts = ["Cold air flows in along the floor", "Cold air flows in along the ceiling",
                "Cold air enters equally everywhere", "Cold air does not enter"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("Cold air flows in along the floor"))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # House (rectangle outline)
        ax.plot([-1.5, 1.5, 1.5, -1.5, -1.5], [-0.7, -0.7, 0.8, 0.8, -0.7],
                color="#212121", linewidth=2)
        # Door (open) on right side
        ax.plot([1.5, 1.5], [-0.7, 0.3], color="#ffffff", linewidth=4)  # door opening
        # Inside heater symbol
        ax.text(-1.2, -0.4, "Heater", fontsize=8, color="#c0392b")
        ax.add_patch(mpatches.Rectangle((-1.4, -0.55), 0.4, 0.3,
                                        facecolor="#ffcdd2", edgecolor="#c0392b", linewidth=1.0))
        # Outside cold air arrows (two: one ceiling-level, one floor-level)
        # Floor arrow (cold flows in along the floor) — show as blue arrow
        ax.annotate("", xy=(0.6, -0.55), xytext=(2.3, -0.55),
                    arrowprops=dict(arrowstyle="->", color="#1976d2", lw=2))
        ax.text(2.4, -0.4, "cold", fontsize=9, color="#1976d2")
        # Ceiling arrow (warm air leaves along ceiling) — orange
        ax.annotate("", xy=(2.3, 0.55), xytext=(0.6, 0.55),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.text(2.4, 0.55, "warm", fontsize=9, color="#c0392b")
        ax.set_xlim(-2, 3)
        ax.set_ylim(-1.0, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(opts))
        q = ("A heated house has its front door opened on a cold winter day. "
             "Wind outside is calm. Which describes how cold air enters the "
             "house? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: cold air is denser than warm air."
        return q, answer, img

    def _gen_convex_mirror(self, rng, cfg, level):
        # Convex mirror gives wider field of view → side-view mirror advantage.
        opts = ["Wider field of view (less blind spot)", "Sharper image",
                "Easier to clean", "Cheaper to manufacture"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("Wider field of view (less blind spot)"))

        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Convex mirror cross-section (arc bulging outward)
        ax.add_patch(mpatches.Arc((0, 0), 1.6, 1.0,
                                  theta1=130, theta2=230,
                                  color="#212121", linewidth=2))
        # Light rays from a wide range
        for theta in (-50, -25, 0, 25, 50):
            x_end = 0.8 * math.cos(math.radians(180 + theta))
            y_end = 0.5 * math.sin(math.radians(180 + theta))
            ax.plot([-2.0, x_end], [theta * 0.02, y_end],
                    color="#1976d2", linewidth=0.8)
        # Reflection arrows behind mirror (single bundle indicating wide angle)
        ax.text(-1.5, 0.8, "Wide angle of view", fontsize=10, color="#c0392b")
        # Eye on the right
        ax.add_patch(mpatches.Circle((1.5, 0), 0.15,
                                     facecolor="#ffffff", edgecolor="#212121", linewidth=1.0))
        ax.add_patch(mpatches.Circle((1.5, 0), 0.05,
                                     facecolor="#212121"))
        ax.text(1.5, -0.30, "viewer", fontsize=9, ha="center")
        ax.set_xlim(-2.5, 2.0)
        ax.set_ylim(-1.0, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(opts))
        q = ("Convex mirrors are used as the side-view mirrors of vehicles. "
             "What is the main advantage over a flat mirror? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: the curvature spreads incoming rays from a wider area into the eye."
        return q, answer, img

    def _gen_fire_door_spring(self, rng, cfg, level):
        # Thicker spring closes the door faster (more force).
        opts = ["Closes faster", "Closes slower",
                "Door does not fully close", "No effect"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("Closes faster"))

        fig, ax = plt.subplots(figsize=(5.5, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Door (vertical rectangle) + frame
        ax.plot([-0.05, -0.05], [-0.8, 0.8], color="#212121", linewidth=2)
        ax.add_patch(mpatches.Rectangle((-0.05, -0.8), 1.0, 1.6,
                                        facecolor="#ffe082", edgecolor="#212121", linewidth=1.4))
        # Hinges
        for hy in (-0.6, 0.6):
            ax.add_patch(mpatches.Rectangle((-0.10, hy - 0.05), 0.10, 0.10,
                                            facecolor="#37474f", edgecolor="#212121"))
        # Spring (thick zig-zag) on left side of door
        n_zigs = 8
        ys = np.linspace(0.3, -0.3, n_zigs * 2 + 1)
        xs = np.array([-0.30 if i % 2 == 0 else -0.50 for i in range(len(ys))])
        ax.plot(xs, ys, color="#c0392b", linewidth=4)  # thick spring
        ax.text(-0.40, 0.5, "thicker spring", fontsize=9, ha="center", color="#c0392b")
        # Floor
        ax.plot([-1.5, 1.5], [-0.85, -0.85], color="#212121", linewidth=1)
        ax.set_xlim(-1.0, 1.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}" for i, o in enumerate(opts))
        q = ("Fire doors are kept closed by a spring. If the spring were "
             "replaced with a thicker (stiffer) one, what would the effect be "
             "on the door? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: a stiffer spring exerts more restoring force."
        return q, answer, img

    def _gen_fire_engine_speed(self, rng, cfg, level):
        # If distance constant, car speed = fire engine speed.
        engine_speed = rng.choice([50, 60, 70])
        opts_vals = [engine_speed, engine_speed + 10, engine_speed - 10]
        rng.shuffle(opts_vals)
        ans_idx = opts_vals.index(engine_speed)
        answer = chr(ord("A") + ans_idx)

        fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Road
        ax.plot([-2.5, 2.5], [-0.4, -0.4], color="#212121", linewidth=2)
        # Fire engine (red truck) on right
        ax.add_patch(mpatches.Rectangle((1.0, -0.25), 1.0, 0.5,
                                        facecolor="#e53935", edgecolor="#212121", linewidth=1.2))
        ax.text(1.5, 0, "Fire engine", fontsize=8, ha="center", va="center", color="#ffffff", fontweight="bold")
        # Car (blue) on left
        ax.add_patch(mpatches.Rectangle((-2.0, -0.20), 0.7, 0.4,
                                        facecolor="#1976d2", edgecolor="#212121", linewidth=1.2))
        ax.text(-1.65, 0, "Car", fontsize=8, ha="center", va="center", color="#ffffff", fontweight="bold")
        # Distance label
        ax.annotate("", xy=(1.0, 0.45), xytext=(-1.3, 0.45),
                    arrowprops=dict(arrowstyle="<->", color="#1976d2", lw=1.0))
        ax.text(-0.15, 0.55, "constant distance", fontsize=8, ha="center", color="#1976d2")
        # Speed annotation
        ax.text(1.5, -0.5, f"{engine_speed} mph", fontsize=10, ha="center", color="#c0392b")
        ax.set_xlim(-2.7, 2.5)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {v} mph"
                             for i, v in enumerate(opts_vals))
        q = (f"On a motorway, a car follows a fire engine moving at "
             f"{engine_speed} mph. The distance between the two vehicles is "
             f"constant. What is the car's speed? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: same speed → no relative motion → constant distance."
        return q, answer, img

    def _gen_cylinder_underwater(self, rng, cfg, level):
        # Cylinder with air; air escapes; tank goes deeper. Best reason:
        # water replaces air → tank gets heavier.
        opts = ["Water replaces escaping air, making the tank heavier",
                "Bubbles push the tank down",
                "Metal density increases under pressure",
                "Bubbles lower water density nearby"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("Water replaces escaping air, making the tank heavier"))

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Water region
        water = mpatches.Rectangle((-1.2, -2.0), 2.4, 2.0,
                                   facecolor="#bbdefb", alpha=0.5, edgecolor="none")
        ax.add_patch(water)
        # Cylinder (vertical) sinking
        ax.add_patch(mpatches.Rectangle((-0.30, -1.5), 0.60, 1.0,
                                        facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.4))
        ax.add_patch(mpatches.Ellipse((0, -0.5), 0.6, 0.10,
                                      facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
        ax.add_patch(mpatches.Ellipse((0, -1.5), 0.6, 0.10,
                                      facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.0))
        # Air bubbles escaping from top
        for (bx, by, br) in [(0.05, -0.40, 0.05), (-0.05, -0.30, 0.04),
                             (0.10, -0.15, 0.05), (0, -0.05, 0.04)]:
            ax.add_patch(mpatches.Circle((bx, by), br,
                                         facecolor="#ffffff", edgecolor="#1976d2", linewidth=1.0))
        # Sinking arrow
        ax.annotate("", xy=(0, -2.0), xytext=(0, -1.7),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.text(-0.50, -1.85, "sinking", fontsize=9, color="#c0392b", ha="right")
        # Water surface line
        ax.plot([-1.2, 1.2], [0, 0], color="#1976d2", linewidth=1.4)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-2.3, 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("A non-pressurised metal cylinder full of air is submerged "
             "underwater. As air slowly escapes through a small opening on "
             "top, the cylinder sinks deeper. What is the best explanation? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: think about how the cylinder's effective mass changes."
        return q, answer, img

    def _gen_pendulum_compare(self, rng, cfg, level):
        # Two pendulums of different string lengths. The shorter swings faster.
        L_short = rng.choice([0.4, 0.5, 0.6])
        L_long = L_short + rng.choice([0.3, 0.4, 0.5])
        # Randomize labels
        layout = rng.choice(["A_short", "B_short"])
        if layout == "A_short":
            la, lb = L_short, L_long
            answer = "A"
        else:
            la, lb = L_long, L_short
            answer = "B"
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Mounting bar
        ax.plot([-1.5, 1.5], [1.0, 1.0], color="#212121", linewidth=3)
        for k in range(15):
            xa = -1.5 + k * 0.21
            ax.plot([xa, xa - 0.10], [1.0, 1.10], color="#212121", linewidth=0.7)
        # Two pendulums
        for cx, L, label in [(-0.7, la, "A"), (0.7, lb, "B")]:
            # String at slight angle (showing as motion blur)
            ax.plot([cx, cx], [1.0, 1.0 - L], color="#212121", linewidth=1.2)
            # Bob
            ax.add_patch(mpatches.Circle((cx, 1.0 - L), 0.10,
                                         facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
            ax.text(cx, 1.0 - L - 0.20, label, fontsize=12, ha="center",
                    fontweight="bold")
        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-0.4, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)
        q = ("Two pendulums hang from a ceiling. They have different string "
             "lengths, but identical bobs. Which pendulum will swing back and "
             "forth faster? "
             "(A) Pendulum A (B) Pendulum B (C) Equal. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: shorter string → shorter period."
        return q, answer, img

    def _gen_centripetal_string(self, rng, cfg, level):
        # Stone on a string spun in a horizontal circle. When the string tears,
        # the stone flies tangentially (perpendicular to the radius at the
        # release point).
        opts = ["Tangent to the circle (perpendicular to the radius)",
                "Toward the center of the circle",
                "Directly outward from the center"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index(
            "Tangent to the circle (perpendicular to the radius)"))

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Circle (stone's path)
        circle = mpatches.Circle((0, 0), 0.7,
                                 facecolor="none", edgecolor="#212121",
                                 linewidth=1.2, linestyle="--")
        ax.add_patch(circle)
        # Hand at center
        ax.add_patch(mpatches.Circle((0, 0), 0.05,
                                     facecolor="#212121"))
        ax.text(0, -0.18, "hand", fontsize=8, ha="center")
        # Stone at top of circle
        ax.add_patch(mpatches.Circle((0, 0.7), 0.07,
                                     facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
        # String from hand to stone (with tear marker)
        ax.plot([0, 0], [0.05, 0.7], color="#212121", linewidth=1.0)
        ax.plot([0.04, -0.04, 0.04, -0.04], [0.30, 0.32, 0.34, 0.36],
                color="#c0392b", linewidth=1.5)
        ax.text(0.20, 0.33, "tear!", fontsize=8, color="#c0392b")
        # Possible flight directions (arrows)
        # tangent (rightward)
        ax.annotate("", xy=(0.7, 0.7), xytext=(0.05, 0.7),
                    arrowprops=dict(arrowstyle="->", color="#1976d2", lw=2))
        ax.text(0.45, 0.85, "tangent", fontsize=9, color="#1976d2")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("A child swings a stone on a string in a circular motion. The "
             "string suddenly tears at the moment shown. In which direction "
             "does the stone fly off? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: with the centripetal force suddenly removed, the stone continues in the direction of its instantaneous velocity (tangent)."
        return q, answer, img

    # =========================================================== #
    # NEW singleton template additions
    # =========================================================== #

    def _gen_belt_rpm_statement(self, rng, cfg, level):
        # Belt drive: 2 pulleys, A drives B via belt. RPM of A and pulley
        # diameters lead to RPM of B = RPM_A * D_A / D_B.
        # 5-option MCQ: 1 correct statement, 4 wrong-direction or wrong-RPM.
        rpm_a = rng.choice([100, 200, 400])
        ratio = rng.choice([0.25, 0.5, 2, 4])  # B size / A size — but RPM is inverse
        # If ratio = D_A / D_B, then RPM_B = RPM_A * (D_A / D_B). Use this directly.
        rpm_b = int(rpm_a * ratio)
        # Belt drive (single belt, both pulleys spin same rotational direction)
        dir_a = rng.choice(["clockwise", "anticlockwise"])
        dir_b = dir_a
        correct_stmt = f"B turns {dir_b} at {rpm_b} RPM"
        wrong_stmts = [
            f"B turns {'anticlockwise' if dir_b == 'clockwise' else 'clockwise'} at {rpm_b} RPM",
            f"B turns {dir_b} at {int(rpm_b * 2)} RPM",
            f"B turns {dir_b} at {max(1, int(rpm_b / 2))} RPM",
            f"A and B turn in opposite directions",
        ]
        rng.shuffle(wrong_stmts)
        opts = [correct_stmt] + wrong_stmts[:4]
        rng.shuffle(opts)
        ans_idx = opts.index(correct_stmt)
        answer = chr(ord("A") + ans_idx)

        # Render: two pulleys with belt
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # A on left
        rA = 0.50
        rB = rA / ratio  # so rA / rB = ratio → RPM relation
        rB = max(0.20, min(0.90, rB))
        ax.add_patch(mpatches.Circle((-1.2, 0), rA,
                                     facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.4))
        ax.text(-1.2, 0, "A", fontsize=14, ha="center", va="center", fontweight="bold")
        ax.add_patch(mpatches.Circle((1.2, 0), rB,
                                     facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.4))
        ax.text(1.2, 0, "B", fontsize=14, ha="center", va="center", fontweight="bold")
        # Belt: two horizontal lines tangent to top and bottom (approx)
        ax.plot([-1.2, 1.2], [rA, rB], color="#212121", linewidth=1.5)
        ax.plot([-1.2, 1.2], [-rA, -rB], color="#212121", linewidth=1.5)
        # Direction arrow on A
        if dir_a == "clockwise":
            arc = mpatches.Arc((-1.2, 0), 2 * 0.3, 2 * 0.3,
                               theta1=90, theta2=350, color="#1976d2", lw=2)
        else:
            arc = mpatches.Arc((-1.2, 0), 2 * 0.3, 2 * 0.3,
                               theta1=350, theta2=90, color="#1976d2", lw=2)
        ax.add_patch(arc)
        # A's RPM annotation
        ax.text(-1.2, -rA - 0.18, f"A: {rpm_a} RPM, {dir_a}",
                fontsize=9, ha="center", va="top")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-1.2, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("Consider the belt-drive system shown. A is the driver pulley "
             f"and B is connected by a belt. Which statement is true? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D/E) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: with a single (uncrossed) belt, both pulleys rotate the same direction; RPM ratio is inverse to diameter ratio."
        return q, answer, img

    def _gen_rack_pinion(self, rng, cfg, level):
        # Two cogwheels with a rack between them. Rack moves; both cogwheels
        # rotate, but in OPPOSITE directions (rack contacts opposite sides).
        # If diameters differ → different angular velocities.
        # Most pedagogically interesting case (reference qid 169): different
        # diameters → "Diff dir, diff vel" = D.
        # Randomize diameters: most often, different sizes → answer D.
        same_size = rng.random() < 0.30
        if same_size:
            answer_text = "Different directions, same velocity"
        else:
            answer_text = "Different directions, different velocities"
        opts = ["Same direction, same velocity",
                "Same direction, different velocities",
                "Different directions, same velocity",
                "Different directions, different velocities"]
        ans_idx = opts.index(answer_text)
        answer = chr(ord("A") + ans_idx)

        # Render
        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Two cogwheels above and below a horizontal rack
        rA = 0.40
        rB = rA if same_size else 0.55
        # Top cogwheel
        ax.add_patch(mpatches.Circle((0, rA + 0.15), rA,
                                     facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.4))
        ax.text(0, rA + 0.15, "A", fontsize=12, ha="center", va="center", fontweight="bold")
        # Bottom cogwheel
        ax.add_patch(mpatches.Circle((0, -rB - 0.15), rB,
                                     facecolor="#bdbdbd", edgecolor="#212121", linewidth=1.4))
        ax.text(0, -rB - 0.15, "B", fontsize=12, ha="center", va="center", fontweight="bold")
        # Rack — horizontal bar (with teeth on top and bottom)
        rack = mpatches.Rectangle((-1.5, -0.10), 3.0, 0.20,
                                  facecolor="#bcaaa4", edgecolor="#212121", linewidth=1.2)
        ax.add_patch(rack)
        # Teeth above (touching A)
        for x in np.linspace(-1.4, 1.4, 14):
            ax.plot([x, x + 0.05, x + 0.10], [0.10, 0.13, 0.10],
                    color="#212121", linewidth=0.8)
        # Direction arrow on rack
        ax.annotate("", xy=(1.7, 0), xytext=(0.4, 0),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.text(1.0, -0.30, "rack moves →",
                fontsize=8, ha="center", color="#c0392b")
        ax.set_xlim(-2.0, 2.2)
        ax.set_ylim(-rB - 0.7, rA + 0.7)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("Two cogwheels A (above) and B (below) are each driven by a "
             "central rack moving horizontally as shown. Considering "
             "rotational direction and angular velocity of the two cogwheels, "
             f"which is correct? {opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: a rack contacts the two cogwheels on opposite sides; angular velocity = rack speed / radius."
        return q, answer, img

    def _gen_seesaw_acrobat(self, rng, cfg, level):
        # Acrobat off-center on a seesaw. To balance, move toward the longer
        # side. We choose a direction → answer letter.
        # We'll show a seesaw with the figure standing on one side (left or right
        # of the fulcrum). To balance: move toward the OTHER side (away from
        # the fulcrum on the heavier-loaded side reduces the imbalance).
        # Actually: if acrobat is on the LEFT of the fulcrum, the left side
        # tilts down. The fulcrum side has more torque. To balance the seesaw
        # (assume something on right), the acrobat needs to move closer to
        # fulcrum (toward the right). For simplicity, assume the seesaw also
        # has a counterweight on the right, and the acrobat needs to move TO
        # the right (toward "B").
        figure_side = rng.choice(["left", "right"])
        # Direction to move = toward fulcrum (to reduce imbalance) if the
        # acrobat side is too heavy; OR away from fulcrum to compensate if
        # the acrobat side is too light. We'll fix: acrobat on the lighter
        # side (counterweight is heavier on the OTHER side), so acrobat must
        # move AWAY from fulcrum (further to that side) to add torque.
        # Direction: same side as figure → move further away from center.
        if figure_side == "left":
            opts = ["Move further left", "Move further right", "Stay still"]
            answer_text = "Move further left"
        else:
            opts = ["Move further left", "Move further right", "Stay still"]
            answer_text = "Move further right"
        answer = chr(ord("A") + opts.index(answer_text))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Fulcrum
        ax.add_patch(mpatches.Polygon([(0, 0), (-0.30, -0.40), (0.30, -0.40)],
                                      closed=True, facecolor="#9e9e9e",
                                      edgecolor="#212121", linewidth=1.0))
        # Seesaw bar — slightly tilted so the heavier counterweight side dips
        # We'll put the figure on the lighter side (raised up).
        if figure_side == "left":
            # Counterweight is on the right → right side dips
            ax.plot([-1.5, 1.5], [0.20, -0.20], color="#212121", linewidth=3)
            # Counterweight (on right, dipped)
            ax.add_patch(mpatches.Rectangle((1.20, -0.20), 0.40, 0.40,
                                            facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
            ax.text(1.40, 0, "W", fontsize=10, ha="center", va="center", fontweight="bold")
            # Acrobat on left side (lifted up)
            ac_x = -1.0
            ac_y = 0.20 - (-1.0 + 1.5) * (0.20 - (-0.20)) / 3.0 + 0.10
        else:
            ax.plot([-1.5, 1.5], [-0.20, 0.20], color="#212121", linewidth=3)
            ax.add_patch(mpatches.Rectangle((-1.60, -0.20), 0.40, 0.40,
                                            facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
            ax.text(-1.40, 0, "W", fontsize=10, ha="center", va="center", fontweight="bold")
            ac_x = 1.0
            ac_y = -0.20 + (1.0 + 1.5) * (0.20 - (-0.20)) / 3.0 + 0.10
        # Stick figure for acrobat
        ax.add_patch(mpatches.Circle((ac_x, ac_y + 0.30), 0.10,
                                     facecolor="#ffe0b2", edgecolor="#212121", linewidth=1.0))
        ax.plot([ac_x, ac_x], [ac_y + 0.20, ac_y - 0.10],
                color="#212121", linewidth=2)
        ax.plot([ac_x, ac_x - 0.10], [ac_y - 0.10, ac_y - 0.30],
                color="#212121", linewidth=2)
        ax.plot([ac_x, ac_x + 0.10], [ac_y - 0.10, ac_y - 0.30],
                color="#212121", linewidth=2)
        # Ground
        ax.plot([-2.0, 2.0], [-0.45, -0.45], color="#212121", linewidth=1)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("An acrobat stands on one side of a seesaw which is unbalanced, "
             "tilting away from him. To balance the seesaw, in which "
             f"direction must the acrobat move along the bar? {opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: torque = weight × distance from the fulcrum; move further from the pivot to add torque on his side."
        return q, answer, img

    def _gen_capacitor_id(self, rng, cfg, level):
        # Pure factual MCQ: which device stores electrical energy?
        # Render a circuit with a generic component shown.
        opts = ["Capacitor", "Resistor", "Diode", "Inductor"]
        rng.shuffle(opts)
        ans_idx = opts.index("Capacitor")
        answer = chr(ord("A") + ans_idx)

        fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Battery on left
        ax.plot([0, 0], [-0.2, 0.2], color="#212121", linewidth=3)
        ax.plot([0.1, 0.1], [-0.4, 0.4], color="#212121", linewidth=2)
        # Wire loop (rectangular)
        ax.plot([0.1, 4.0], [0.4, 0.4], color="#212121", linewidth=1.5)
        ax.plot([4.0, 4.0], [0.4, -0.4], color="#212121", linewidth=1.5)
        ax.plot([0.1, 4.0], [-0.4, -0.4], color="#212121", linewidth=1.5)
        # Capacitor symbol on top wire (two parallel plates) — generic,
        # purpose is recall not visual identification.
        ax.plot([2.0, 2.0], [0.30, 0.50], color="#212121", linewidth=2.4)
        ax.plot([2.10, 2.10], [0.30, 0.50], color="#212121", linewidth=2.4)
        ax.text(2.05, 0.65, "?", fontsize=14, ha="center", color="#c0392b", fontweight="bold")
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.8, 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("A device used to store electrical energy in a circuit (such as "
             "the component marked '?') is called a: "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: this component stores energy in an electric field between two plates."
        return q, answer, img

    def _gen_balls_falling(self, rng, cfg, level):
        # Two balls, one launched horizontally from height H, one dropped
        # from same height. Both land at the same time (gravity equal).
        opts = ["Both land at the same time",
                "Ball A (horizontal launch) lands first",
                "Ball B (dropped) lands first"]
        rng.shuffle(opts)
        answer = chr(ord("A") + opts.index("Both land at the same time"))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Cliff edge
        ax.add_patch(mpatches.Rectangle((-1.5, -0.6), 1.5, 1.0,
                                        facecolor="#bcaaa4", edgecolor="#212121", linewidth=1.0))
        # Ball A (horizontal launch, parabolic trajectory)
        xs = np.linspace(-0.05, 1.5, 30)
        ys = 0.4 - 0.4 * (xs + 0.05) ** 2 / 1.0
        ax.plot(xs, ys, color="#1976d2", linewidth=1.0, linestyle="--")
        ax.add_patch(mpatches.Circle((-0.05, 0.4), 0.08,
                                     facecolor="#1976d2", edgecolor="#212121"))
        ax.text(-0.05, 0.55, "A", fontsize=10, ha="center", fontweight="bold",
                color="#1976d2")
        # Velocity arrow horizontal
        ax.annotate("", xy=(0.3, 0.4), xytext=(0, 0.4),
                    arrowprops=dict(arrowstyle="->", color="#1976d2", lw=1.5))
        # Ball B (dropped from same height)
        ax.add_patch(mpatches.Circle((-1.0, 0.4), 0.08,
                                     facecolor="#c0392b", edgecolor="#212121"))
        ax.text(-1.0, 0.55, "B", fontsize=10, ha="center", fontweight="bold",
                color="#c0392b")
        ax.plot([-1.0, -1.0], [0.4, -0.6], color="#c0392b",
                linewidth=1.0, linestyle="--")
        # Ground
        ax.plot([-2, 2.5], [-0.6, -0.6], color="#212121", linewidth=1)
        ax.set_xlim(-2, 2.5)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("Two identical balls are released from the same height. Ball A "
             "is launched horizontally from the edge of a cliff at high "
             "speed; Ball B is simply dropped from rest beside it. Ignoring "
             "air resistance, which ball lands first? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: vertical motion is independent of horizontal motion under gravity."
        return q, answer, img

    def _gen_pendulum_wagon(self, rng, cfg, level):
        # Pendulum on accelerating wagon: pendulum tilts BACKWARD if wagon
        # accelerates FORWARD (inertia). So if pendulum tilts backward, the
        # wagon must be accelerating forward.
        tilt_dir = rng.choice(["backward", "forward"])
        if tilt_dir == "backward":
            # wagon accelerating forward (in image: rightward)
            answer_text = "Wagon accelerating to the right"
        else:
            answer_text = "Wagon accelerating to the left"
        opts = ["Wagon accelerating to the left",
                "Wagon accelerating to the right",
                "Wagon at constant velocity"]
        ans_idx = opts.index(answer_text)
        answer = chr(ord("A") + ans_idx)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Wagon (rectangle on wheels)
        wagon = mpatches.Rectangle((-1.0, -0.10), 2.0, 0.40,
                                   facecolor="#a1887f", edgecolor="#212121", linewidth=1.4)
        ax.add_patch(wagon)
        # Wheels
        for wx in (-0.7, 0.7):
            ax.add_patch(mpatches.Circle((wx, -0.10), 0.12,
                                         facecolor="#37474f", edgecolor="#212121", linewidth=1.0))
        # Pendulum mounting on top center
        ax.add_patch(mpatches.Rectangle((-0.05, 0.30), 0.10, 0.10,
                                        facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
        # Pendulum at angle (tilted back or forward depending on tilt_dir)
        if tilt_dir == "backward":
            # Pendulum bob at slight angle to the LEFT
            theta = math.radians(25)
            bob_x = 0 - 0.7 * math.sin(theta)
            bob_y = 0.40 - 0.7 * math.cos(theta)
        else:
            theta = math.radians(-25)
            bob_x = 0 - 0.7 * math.sin(theta)
            bob_y = 0.40 - 0.7 * math.cos(theta)
        ax.plot([0, bob_x], [0.40, bob_y], color="#212121", linewidth=1.2)
        ax.add_patch(mpatches.Circle((bob_x, bob_y), 0.10,
                                     facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.0))
        # Ground
        ax.plot([-2.0, 2.0], [-0.30, -0.30], color="#212121", linewidth=1)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-0.7, 1.0)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("A pendulum hangs from the top of a wagon. The pendulum is held "
             "at the angle shown — not vertical. Which describes the wagon's "
             f"motion? {opt_block}. "
             "Reply with the letter (A/B/C) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: the pendulum tilts opposite to the wagon's acceleration (inertia)."
        return q, answer, img

    def _gen_bolt_cutter(self, rng, cfg, level):
        # 5-option MCQ from reference qid 420: tight nut + spanner.
        # Easiest = LONG spanner held at the FAR end of the handle (= maximum
        # mechanical advantage).
        opts = ["Long spanner held at the far end (longest moment arm)",
                "Long spanner held near the nut",
                "Short spanner held at the far end",
                "Short spanner held near the nut",
                "Tight-grip spanner with both hands"]
        rng.shuffle(opts)
        ans_idx = opts.index("Long spanner held at the far end (longest moment arm)")
        answer = chr(ord("A") + ans_idx)

        fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Hex nut at center
        ax.add_patch(mpatches.RegularPolygon((0, 0), numVertices=6, radius=0.20,
                                             orientation=math.radians(30),
                                             facecolor="#9e9e9e", edgecolor="#212121", linewidth=1.4))
        # Spanner shaft (long)
        ax.plot([0.18, 1.5], [0, 0], color="#37474f", linewidth=4)
        # Force arrow at far end
        ax.annotate("", xy=(1.5, -0.4), xytext=(1.5, 0.0),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2))
        ax.text(1.5, -0.55, "Force", fontsize=9, ha="center", color="#c0392b")
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.9, 0.6)
        ax.set_aspect("equal")
        ax.axis("off")
        img = self.fig_to_pil(fig, dpi=120)

        opt_block = " ".join(f"({chr(ord('A') + i)}) {o}"
                             for i, o in enumerate(opts))
        q = ("A nut is very tight. Which scenario requires the least effort "
             "to loosen it using a spanner? "
             f"{opt_block}. "
             "Reply with the letter (A/B/C/D/E) in <answer>...</answer>.")
        if cfg["give_hint"]:
            q = q + " Hint: longer lever arm (further from the nut) means more torque for the same applied force."
        return q, answer, img


if __name__ == "__main__":
    import os
    import sys
    out_dir = "/tmp/env_check_bennett"
    os.makedirs(out_dir, exist_ok=True)

    # Mode 1: smoke test every sub-template by direct invocation.
    if len(sys.argv) > 1 and sys.argv[1] == "subs":
        env = BennettMechanicalQA()
        env.seed = 42
        env.parameter = {"level": 6}
        env._rng = random.Random(42)
        cfg = env._level_config(6)
        rng = random.Random(42)
        n_pass = 0
        n_fail = 0
        for sub in _SUB_TEMPLATES:
            try:
                func_name = f"_gen_{sub}"
                # Find by exact name
                fn = getattr(env, func_name, None)
                if fn is None:
                    print(f"  {sub}: NO_DISPATCH ({func_name} not found)")
                    n_fail += 1
                    continue
                result = fn(rng, cfg, 6)
                if result is None:
                    print(f"  {sub}: returned None")
                    n_fail += 1
                    continue
                q, a, img = result
                # Save image
                img.save(os.path.join(out_dir, f"sub_{sub}.png"))
                # Verify with wrapper
                env._answer = a
                env._question = q
                env._image = img
                v = env.verify(f"<answer>{a}</answer>")
                v2 = env.verify(a)
                print(f"  {sub}: ans={a} verify={v['accuracy']} bare={v2['accuracy']}")
                if v["accuracy"] == 1:
                    n_pass += 1
                else:
                    n_fail += 1
            except Exception as e:
                import traceback
                print(f"  {sub}: EXCEPTION {type(e).__name__}: {e}")
                traceback.print_exc()
                n_fail += 1
        print(f"\nSubtemplate smoke: {n_pass}/{n_pass + n_fail} pass")
        sys.exit(0)

    # Mode 2: standard sample test
    env = BennettMechanicalQA()
    for level in (0, 3, 6, 9):
        for seed in range(5):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']}")
