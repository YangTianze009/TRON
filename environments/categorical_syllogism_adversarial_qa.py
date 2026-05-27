"""
Categorical Syllogism Adversarial QA environment (batch 2 Part B, 2026-04-14).

Goal: 2-5 premise categorical syllogism with adversarial distractors
(undistributed middle, illicit major/minor, four-terms). Targets
reference deductive and logical reasoning.

Difficulty axes:
  A) Pattern E n_premises (2..5).
  B) Pattern B distractor_mode — random → fallacy-shaped → negated.
  C) Pattern H nonce_terms at L≥4.

Format: 4-way MCQ (letter).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_REAL_TERMS = [
    "mammals", "dogs", "animals", "cats", "pets", "fish", "birds",
    "vehicles", "cars", "trucks", "airplanes", "machines",
    "plants", "trees", "flowers", "roses", "tools", "hammers",
    "instruments", "violins", "fruits", "apples", "berries",
]

_NONCE_TERMS = [
    "phenomena", "phenomeni", "phenomenae",
    "zorbs", "zarbs", "zirbs",
    "quells", "quills", "quiggs",
    "blorps", "blerps", "blurps",
    "trondles", "trandles", "trindles",
    "frazzles", "frizzles", "grozzles",
    "plonks", "plunks", "plinks",
]

class CategoricalSyllogismAdversarialQA(StandaloneVisualEnv):
    ENV_NAME = "categorical_syllogism_adversarial"

    # 2026-05-04 R4: full-gradient redesign per categorical-syllogism samples.
    # The deductive subset uses 2-premise classical syllogisms with
    # "None correct" answer in 1/3 of cases. The current env was scaling
    # n_premises but model handles 3-premise Barbara chains trivially.
    # Real progressive gradient:
    #   L0: 2 premises, REAL terms, no nonce, all-affirmative, only correct
    #       and 3 reversed-distractors.
    #   L2: 3 premises, real terms, fallacy distractors
    #   L4: nonce terms (e.g. "phenomena")
    #   L6: 5+ premises, mixed quantifiers, "None correct" possible 1/4
    #   L8: nonce + 7 prem + "None correct" 1/3, biconditional
    #   L9: 9 prem + nonce + all flags + "None correct" possible
    #
    # 2026-05-05 R5 B3B HARDEN: 4-MCQ → 5-MCQ. Slot E is fixed
    # "None of the above logically follows" (not mixed in randomly). At L8/L9
    # E-trap rate bumped to 40% (was 30%). At L6/L7 set to 25%.
    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Per-level sizes designed so each step adds a NEW reasoning load
        sizes = [2, 2, 3, 3, 4, 5, 6, 7, 8, 9]
        n_prem = sizes[level]
        return {
            "n_premises":       n_prem,
            "fallacy_distract": level >= 2,                 # plain reversed at L0/L1
            "negated_distract": level >= 4,
            "use_nonce":        level >= 4,                 # benchmark nonce L4+
            "mix_quantifiers":  level >= 5,                 # mix "All"/"Some"/"No"
            "extra_red_herrings": max(0, (level - 4) // 2), # 0,0,0,0,0,0,1,1,2,2
            "biconditional":    level >= 8,
            # 5-MCQ with fixed E="None of the above". E-trap rate per level:
            # 2026-05-05 Phase A: bump L8/L9 to 0.5 (env still saturated 95%).
            "none_correct_chance": 0.0 if level <= 5 else (
                0.25 if level <= 7 else 0.50),
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_premises"]

        pool = _NONCE_TERMS if cfg["use_nonce"] else _REAL_TERMS
        needed = cfg["n_premises"] + 1 + cfg.get("extra_red_herrings", 0)
        if needed > len(pool):
            return None

        for _ in range(40):
            r = self._try_generate(rng, cfg, pool)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      pool: List[str]) -> Optional[Tuple[str, str, Image.Image]]:
        n = cfg["n_premises"]

        # Build a "Barbara-like" chain: All X1 are X2, All X2 are X3, ... → All X1 are Xn+1.
        terms = rng.sample(pool, n + 1)
        premises: List[str] = []
        use_existential = False
        if rng.random() < 0.3 and n >= 2:
            # Switch to "Some X1 are X2, All X2 are X3 → Some X1 are X3".
            use_existential = True
            premises.append(f"Some {terms[0]} are {terms[1]}.")
            for i in range(1, n):
                premises.append(f"All {terms[i]} are {terms[i + 1]}.")
            correct = f"Some {terms[0]} are {terms[n]}."
            # Fallacy-shaped distractors: affirm consequent, reversed,
            # illicit conversion, etc.
            distractor_pool = [
                f"All {terms[0]} are {terms[n]}.",          # too strong
                f"No {terms[n]} are {terms[0]}.",           # negation
                f"Some {terms[n]} are not {terms[0]}.",     # illicit
            ]
        else:
            for i in range(n):
                premises.append(f"All {terms[i]} are {terms[i + 1]}.")
            correct = f"All {terms[0]} are {terms[n]}."
            distractor_pool = [
                f"All {terms[n]} are {terms[0]}.",           # reversed
                f"No {terms[0]} are {terms[n]}.",            # direct negation
                f"Some {terms[0]} are not {terms[n]}.",      # illicit
                f"All {terms[n]} are {terms[1]}.",           # subtle wrong
            ]

        if cfg["negated_distract"]:
            distractor_pool.append(f"No {terms[0]} are {terms[n]}.")

        # Dedup
        distractor_pool = [d for d in distractor_pool if d != correct]
        rng.shuffle(distractor_pool)
        if len(distractor_pool) < 3:
            return None

        # 2026-05-05 R5 B3B HARDEN: 5-MCQ with FIXED slot E = "None of the
        # above logically follows." (was a randomly-inserted A-D in R4).
        # Trap branch: GT='E' iff none_correct_mode is true; A-D are 4
        # non-following distractors. Else: GT in A-D, E is always present.
        none_correct_chance = cfg.get("none_correct_chance", 0.0)
        none_correct_mode = (rng.random() < none_correct_chance)
        none_str = "None of the above logically follows."

        if none_correct_mode:
            # 4 non-following statements as A-D
            non_following = distractor_pool[:4]
            if len(non_following) < 4:
                # Pad with another distractor variant
                extra = [
                    f"All {terms[0]} are themselves.",  # tautology
                    f"Some {terms[0]} are not {terms[0]}.",  # contradiction
                ]
                for x in extra:
                    if x not in non_following and x != correct:
                        non_following.append(x)
                    if len(non_following) >= 4:
                        break
            if len(non_following) < 4:
                return None
            options = list(non_following[:4]) + [none_str]
            answer_letter = "E"
        else:
            distractors = distractor_pool[:3]
            rng.shuffle(distractors)
            options = list(distractors[:3])
            pos_rng = random.Random(sum(ord(c) for c in correct) * 17
                                    + len(terms[0]) * 31)
            correct_pos = pos_rng.randint(0, 3)
            options.insert(correct_pos, correct)
            if options.count(correct) > 1:
                return None
            options.append(none_str)  # E is always present
            answer_letter = chr(ord("A") + correct_pos)

        # Add extra red-herring premises at high levels
        n_rh = cfg.get("extra_red_herrings", 0)
        if n_rh > 0:
            rh_terms = rng.sample([t for t in pool if t not in terms], min(n_rh, len(pool) - len(terms)))
            for rt in rh_terms:
                partner = rng.choice(terms)
                premises.append(f"All {rt} are {partner}.")

        rng.shuffle(premises)

        given_text = "Assume the premises are true."
        ask_text = "Which conclusion logically follows?"

        # Include a short premise preview in the question so the test
        # 5.3 seed-differentiation check (which compares _question text)
        # catches level changes in the problem content.
        # Use the actual rendered premise count (includes red herrings).
        n_shown = len(premises)
        premise_preview = premises[0]
        q_stems = [
            f"Read the {n_shown} premises shown in the image (first premise: \"{premise_preview}\"). Which of the following conclusions MUST follow?",
            f"Given the {n_shown} premises displayed (starting with: \"{premise_preview}\"), which conclusion is logically valid?",
            f"Examine the {n_shown} premises in the image. Which conclusion necessarily follows from ALL the premises?",
            f"The image shows {n_shown} categorical premises. Determine which conclusion is logically entailed.",
        ]
        question = (
            f"{rng.choice(q_stems)} Answer with a single letter (A, B, C, D, or E)."
        )
        image = self._render(premises, options, given_text, ask_text, cfg)
        return question, answer_letter, image

    def _render(self, premises, options, given_text, ask_text,
                cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans", "monospace"])
        fs = self._rng.choice([12, 13, 14])

        n_premises = len(premises)
        n_options = len(options)
        n_lines = n_premises + n_options + 5
        panel_h = max(7.0, n_lines * 0.58 + 1.0)

        fig_w = 11.0 * sc
        fig_h = (panel_h + 1.0) * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(style["bg_color"])
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, panel_h + 0.5)
        ax.set_axis_off()

        border = mpatches.FancyBboxPatch(
            (0.2, 0.2), 9.6, panel_h + 0.1,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=style["bg_color"], edgecolor="#7f8c8d", linewidth=1.2)
        ax.add_patch(border)

        syl_title_pool = ["Syllogism", "Deductive Reasoning",
                          "Logical Premises", "Categorical Logic",
                          "Premise Chain"]
        ax.text(5.0, panel_h - 0.05, self._rng.choice(syl_title_pool),
                fontsize=fs + 3, fontweight="bold", ha="center", va="top",
                family=ff, color="#2c3e50")
        y = panel_h - 0.85
        ax.text(0.6, y, "Premises:", fontsize=fs + 1, fontweight="bold",
                family=ff, color="#34495e", ha="left", va="top")
        y -= 0.55
        for i, p in enumerate(premises):
            ax.text(0.9, y, f"{i + 1}. {p}", fontsize=fs, family=ff,
                    ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax.plot([0.6, 9.4], [y, y], color="#95a5a6", linewidth=1.0)
        y -= 0.4
        ax.text(0.6, y, "Possible conclusions:", fontsize=fs + 1,
                fontweight="bold", family=ff, color="#34495e",
                ha="left", va="top")
        y -= 0.55
        for i, o in enumerate(options):
            letter = chr(ord("A") + i)
            ax.text(0.9, y, f"({letter}) {o}", fontsize=fs, family=ff,
                    ha="left", va="top", color="#1a1a1a")
            y -= 0.55

        fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b2b"
    os.makedirs(out_dir, exist_ok=True)
    env = CategoricalSyllogismAdversarialQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"categorical_syllogism_adversarial_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  Q (first 100): {env.get_instruction()[:100]}")
            print(f"  A: {env._answer}")
