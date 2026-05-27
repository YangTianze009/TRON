"""
Medication / Ailment Deduction QA — reference L122/L126-style deductive
reasoning over a small text-rendered table of rules.

Style: PURE-OCR pages — no diagrams. The "image" is a paragraph of natural-
language premises rendered as text (matplotlib ax.text block). Statements
involve multi-step arithmetic deduction:
  - "Ailment A lasts 12 days, Ailment B lasts 32 days, ..."
  - "Medicine X halves the duration of Ailment Y"
  - "Medicine Z reduces the duration of Ailment W by N days"
The model OCRs the rules + the candidate statement, applies the chain, and
answers True/False/Insufficient.

Single-letter MCQ answer (A/B/C).

Sample target style (from design notes idx=47..50):
  *idx=47* — "Taking Medicine B will reduce duration of Ailment C by 12 days.
              (A)T (B)F (C)Insufficient." Ans: B.
              Reasoning: C = 32 days; Medicine B halves to 16, so reduces
              by 16 not 12.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_AILMENT_NAMES = ["Ailment A", "Ailment B", "Ailment C", "Ailment D", "Ailment E"]
_MEDICINE_NAMES = ["Medicine A", "Medicine B", "Medicine C", "Medicine D"]


_QUESTION_TEMPLATES_REDUCE_BY = [
    "Taking {med} will reduce the duration of {ail} by {n} days.",
    "{med} reduces the duration of {ail} by {n} days when taken.",
    "If a patient takes {med}, the duration of {ail} is reduced by {n} days.",
]
_QUESTION_TEMPLATES_REDUCE_TO = [
    "Taking {med} will reduce the duration of {ail} to {n} days.",
    "After {med}, the duration of {ail} is {n} days.",
    "{med} makes the duration of {ail} equal to {n} days.",
]
_QUESTION_TEMPLATES_LASTS = [
    "{ail} lasts for {n} days.",
    "The duration of {ail} is {n} days.",
    "Without medicine, {ail} lasts {n} days.",
]
_QUESTION_TEMPLATES_COMPARE = [
    "Taking the appropriate medicine, {ail1} lasts longer than {ail2}.",
    "With appropriate medicine, the duration of {ail1} exceeds that of {ail2}.",
    "After medicine, {ail1} is longer than {ail2}.",
]


_INSTRUCTION_OPTIONS = [
    "(A) True (B) False (C) Insufficient information.",
    "(A) T (B) F (C) Insufficient.",
    "(A) True (B) False (C) Insufficient.",
]
_INSTRUCTION_TAILS = [
    "Reply with the letter (A/B/C) in <answer>...</answer>.",
    "Letter (A/B/C) in <answer>...</answer>.",
    "Place the single letter (A/B/C) in <answer>X</answer>.",
]


class MedicationDeductionQA(StandaloneVisualEnv):
    ENV_NAME = "medication_deduction"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0-L1: 2 ailments, 1 medicine, simple "lasts N days" facts.
        # L2-L3: 2-3 ailments, 1-2 medicines, single-rule deduction.
        # L4-L5: 3 ailments, 2 medicines, mixed rule types.
        # L6+:   3-4 ailments, 2-3 medicines, comparative + chained.
        if level <= 1:
            return {"n_ail": 2, "n_med": 1, "modes": ["lasts", "reduce_by"], "insufficient_rate": 0.15}
        if level <= 3:
            return {"n_ail": 3, "n_med": 2, "modes": ["lasts", "reduce_by", "reduce_to"], "insufficient_rate": 0.25}
        if level <= 5:
            return {"n_ail": 3, "n_med": 2, "modes": ["lasts", "reduce_by", "reduce_to", "compare"], "insufficient_rate": 0.30}
        if level <= 7:
            return {"n_ail": 4, "n_med": 3, "modes": ["lasts", "reduce_by", "reduce_to", "compare"], "insufficient_rate": 0.30}
        return {"n_ail": 4, "n_med": 3, "modes": ["lasts", "reduce_by", "reduce_to", "compare"], "insufficient_rate": 0.30}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7717 + level * 191 + 13)

        # Build durations (each ailment lasts D days, simple integers).
        n_ail = cfg["n_ail"]
        n_med = cfg["n_med"]
        durations: Dict[str, int] = {}
        for i in range(n_ail):
            ail = _AILMENT_NAMES[i]
            durations[ail] = rng.choice([8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32, 36, 40])

        # Build medicine effects: each medicine acts on one ailment (or none — for
        # insufficient-info question generation).
        # effect = ("halves", target_ail) or ("reduce_by", target_ail, n_days) or
        #          ("reduce_to", target_ail, n_days_resulting)
        # For ailments with no medicine listed → "Insufficient" deductions are valid.
        effects: List[Tuple] = []
        # Ensure at least one medicine has an effect if we have any.
        ailments_with_effects = set()
        for j in range(n_med):
            med = _MEDICINE_NAMES[j]
            # Sometimes leave a medicine without a listed rule (rare, only at L4+)
            if level >= 4 and rng.random() < 0.10 and len(effects) > 0:
                continue
            target_ail = _AILMENT_NAMES[rng.randint(0, n_ail - 1)]
            ailments_with_effects.add(target_ail)
            etype = rng.choices(
                ["halves", "reduce_by", "reduce_to"],
                weights=[3, 4, 3], k=1)[0]
            d = durations[target_ail]
            if etype == "halves":
                if d % 2 == 0:
                    effects.append((med, "halves", target_ail))
                else:
                    # Pick a different mode
                    n_days = rng.randint(2, max(3, d // 3))
                    effects.append((med, "reduce_by", target_ail, n_days))
            elif etype == "reduce_by":
                # reduce_by N where N is a small fraction of d
                n_days = rng.randint(2, max(3, d // 3))
                effects.append((med, "reduce_by", target_ail, n_days))
            else:  # reduce_to
                # reduce_to where reduce-to value is in (0, d)
                if d > 6:
                    new_d = rng.choice([d - 4, d - 5, d - 6, d // 2, max(1, d - 8)])
                    new_d = max(1, new_d)
                    if new_d == d:
                        new_d = d - 2
                    effects.append((med, "reduce_to", target_ail, new_d))
                else:
                    effects.append((med, "reduce_by", target_ail, max(1, d // 2)))

        # Compose passage describing facts.
        lines = []
        for ail in _AILMENT_NAMES[:n_ail]:
            d = durations[ail]
            lines.append(f"{ail} lasts {d} days without treatment.")
        for eff in effects:
            med = eff[0]
            etype = eff[1]
            target = eff[2]
            if etype == "halves":
                lines.append(f"{med} halves the duration of {target}.")
            elif etype == "reduce_by":
                n_days = eff[3]
                lines.append(f"{med} reduces the duration of {target} by {n_days} days.")
            else:
                n_days = eff[3]
                lines.append(f"{med} reduces the duration of {target} to {n_days} days.")
        rng.shuffle(lines)

        # Now generate a candidate statement & verifiable truth value.
        # Pick a question mode randomly from cfg["modes"].
        for attempt in range(20):
            mode = rng.choice(cfg["modes"])

            # Decide whether to generate a True / False / Insufficient instance.
            # Insufficient requires asking about an ailment-medicine pair where
            # no listed rule covers (ailment may have no medicine, OR question
            # asks about a medicine and a different ailment).
            target_truth = rng.choices(["T", "F", "I"], weights=[4, 4, 3], k=1)[0]

            stmt, expected = self._build_statement(
                mode, durations, effects, target_truth, rng,
                n_ail=n_ail, n_med=n_med, ailments_with_effects=ailments_with_effects)
            if stmt is None:
                continue
            break
        else:
            return None

        # Compose final question.
        passage = "\n".join(lines)
        ans_letter = {"T": "A", "F": "B", "I": "C"}[expected]
        opt_str = rng.choice(_INSTRUCTION_OPTIONS)
        tail = rng.choice(_INSTRUCTION_TAILS)
        question = (
            f"Read the rules in the image carefully. Decide whether the following "
            f"statement is True, False, or cannot be determined from the rules.\n\n"
            f"Statement: {stmt}\n\n"
            f"Options: {opt_str} {tail}"
        )

        img = self._render_passage(passage, rng)
        return question, ans_letter, img

    # ---------------------------------------------------------------- #

    def _resolve_duration_with_med(
        self, ail: str, med: str, durations: Dict[str, int],
        effects: List[Tuple],
    ) -> Optional[int]:
        """Resolve the duration of ailment when patient takes given medicine.
        Returns None if no rule covers this (med, ail) pair (= Insufficient)."""
        d = durations[ail]
        for eff in effects:
            if eff[0] != med:
                continue
            if eff[2] != ail:
                continue
            if eff[1] == "halves":
                return d // 2
            if eff[1] == "reduce_by":
                return max(0, d - eff[3])
            if eff[1] == "reduce_to":
                return eff[3]
        return None

    def _resolve_min_duration(
        self, ail: str, durations: Dict[str, int], effects: List[Tuple],
    ) -> Optional[int]:
        """Return the minimum duration achievable with any medicine for an
        ailment. None if no medicine targets the ailment."""
        d = durations[ail]
        best = None
        for eff in effects:
            if eff[2] != ail:
                continue
            if eff[1] == "halves":
                v = d // 2
            elif eff[1] == "reduce_by":
                v = max(0, d - eff[3])
            else:
                v = eff[3]
            if best is None or v < best:
                best = v
        return best

    def _build_statement(
        self, mode: str, durations: Dict[str, int], effects: List[Tuple],
        target_truth: str, rng: random.Random, *, n_ail: int, n_med: int,
        ailments_with_effects: set,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Construct (statement_text, expected_letter) where expected ∈ {T,F,I}.
        Returns (None, None) if generation fails."""

        def _pick_med() -> str:
            return _MEDICINE_NAMES[rng.randint(0, n_med - 1)]

        def _pick_ail() -> str:
            return _AILMENT_NAMES[rng.randint(0, n_ail - 1)]

        if mode == "lasts":
            # "Ailment X lasts N days." (without medicine)
            ail = _pick_ail()
            true_d = durations[ail]
            if target_truth == "T":
                n = true_d
                expected = "T"
            elif target_truth == "F":
                n = true_d + rng.choice([-4, -2, 2, 4, 6])
                if n == true_d:
                    n = true_d + 2
                if n <= 0:
                    n = true_d + 2
                expected = "F"
            else:  # Insufficient (we always know untreated duration, so skip)
                return None, None
            tmpl = rng.choice(_QUESTION_TEMPLATES_LASTS)
            return tmpl.format(ail=ail, n=n), expected

        if mode == "reduce_by":
            # "Taking Medicine M will reduce the duration of Ailment A by N days."
            med = _pick_med()
            ail = _pick_ail()
            new_d = self._resolve_duration_with_med(ail, med, durations, effects)
            if new_d is None:
                # No rule for (med, ail). Insufficient if target_truth == I.
                if target_truth == "I":
                    n = rng.randint(2, max(3, durations[ail] // 2))
                    tmpl = rng.choice(_QUESTION_TEMPLATES_REDUCE_BY)
                    return tmpl.format(med=med, ail=ail, n=n), "I"
                return None, None
            true_reduce = durations[ail] - new_d
            if target_truth == "T":
                n = true_reduce
                expected = "T"
            elif target_truth == "F":
                n = true_reduce + rng.choice([-4, -3, -2, 2, 3, 4])
                if n == true_reduce or n <= 0:
                    n = true_reduce + 3
                expected = "F"
            else:
                # Hard insufficient on reduce_by — skip
                return None, None
            tmpl = rng.choice(_QUESTION_TEMPLATES_REDUCE_BY)
            return tmpl.format(med=med, ail=ail, n=n), expected

        if mode == "reduce_to":
            # "Taking Medicine M will reduce the duration of Ailment A to N days."
            med = _pick_med()
            ail = _pick_ail()
            new_d = self._resolve_duration_with_med(ail, med, durations, effects)
            if new_d is None:
                if target_truth == "I":
                    n = rng.randint(2, max(3, durations[ail] - 1))
                    tmpl = rng.choice(_QUESTION_TEMPLATES_REDUCE_TO)
                    return tmpl.format(med=med, ail=ail, n=n), "I"
                return None, None
            if target_truth == "T":
                n = new_d
                expected = "T"
            elif target_truth == "F":
                n = new_d + rng.choice([-3, -2, -1, 1, 2, 3])
                if n == new_d or n <= 0:
                    n = new_d + 2
                expected = "F"
            else:
                return None, None
            tmpl = rng.choice(_QUESTION_TEMPLATES_REDUCE_TO)
            return tmpl.format(med=med, ail=ail, n=n), expected

        if mode == "compare":
            # "Taking appropriate medicine, ail1 lasts longer than ail2."
            ail1 = _pick_ail()
            ail2 = _pick_ail()
            if ail1 == ail2:
                return None, None
            min1 = self._resolve_min_duration(ail1, durations, effects)
            min2 = self._resolve_min_duration(ail2, durations, effects)
            d1 = min1 if min1 is not None else durations[ail1]
            d2 = min2 if min2 is not None else durations[ail2]
            true_value = d1 > d2
            if target_truth == "T":
                # Generate a True comparison: pick (ail1, ail2) such that d1 > d2
                if true_value:
                    expected = "T"
                else:
                    # swap
                    ail1, ail2 = ail2, ail1
                    expected = "T"
                tmpl = rng.choice(_QUESTION_TEMPLATES_COMPARE)
                return tmpl.format(ail1=ail1, ail2=ail2), expected
            if target_truth == "F":
                if not true_value:
                    expected = "F"
                else:
                    ail1, ail2 = ail2, ail1
                    expected = "F"
                tmpl = rng.choice(_QUESTION_TEMPLATES_COMPARE)
                return tmpl.format(ail1=ail1, ail2=ail2), expected
            # Insufficient: requires both ailments NOT to have any medicine →
            # but we always know untreated duration for both ailments, so
            # comparison is always determinate. Skip.
            return None, None

        return None, None

    # ---------------------------------------------------------------- #

    def _render_passage(self, passage: str, rng: random.Random) -> Image.Image:
        """Render a passage of text as an image (reference 'text-in-image' style)."""
        n_lines = passage.count("\n") + 1
        # Width auto-scales with longest line.
        max_line_len = max(len(l) for l in passage.split("\n"))
        # Approx width: 0.10 inches per char + margin
        fig_w = max(5.0, min(11, 0.085 * max_line_len + 1.0))
        fig_h = max(2.5, 0.55 * n_lines + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Title
        title_y = 0.96
        ax.text(0.5, title_y, "Rules", fontsize=14, ha="center", va="top",
                fontweight="bold", color="#212121")

        # Body lines
        line_y = 0.88
        line_step = 0.78 / max(n_lines, 1)
        for i, line in enumerate(passage.split("\n")):
            ax.text(0.05, line_y - i * line_step, line, fontsize=12,
                    ha="left", va="top", color="#212121",
                    family="DejaVu Sans")
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_meddeduct"
    os.makedirs(out_dir, exist_ok=True)
    env = MedicationDeductionQA()
    for level in (0, 3, 6, 9):
        for seed in range(5):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            v2 = env.verify(env._answer)
            v3 = env.verify("<answer>Z</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']} bare={v2['accuracy']} wrong={v3['accuracy']}")
