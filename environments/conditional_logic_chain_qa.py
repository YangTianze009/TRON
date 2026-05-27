"""
Conditional Logic Chain QA environment (batch 2 Part B, 2026-04-14).

Goal: multi-hop if-then chains. Given rules and starting facts, conclude
which of the candidate statements must be true. Targets reference
deductive, logical reasoning.

Difficulty axes:
  A) Pattern E chain_length (2..7).
  B) Pattern B n_red_herrings and adversarial distractors.
  C) Pattern A use_contrapositive at L≥4.

Format: 4-way MCQ (letter). Yes/No/Cannot options at all levels.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PROP_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
    "Eta", "Theta", "Iota", "Kappa", "Lambda",
]

class ConditionalLogicChainQA(StandaloneVisualEnv):
    ENV_NAME = "conditional_logic_chain"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Pool (_PROP_NAMES) has 11 names, so chain+1+red_herrings <= 11.
        # Previous config asked for 10+1+4 = 15 at L9 — generation always
        # failed, causing the 0.20 pass-rate floor. Also contrapositive was
        # enabled from L0 which pushed L0 down to 0.30.
        # New schedule (3→8 chain, 0→3 red) respects pool size and keeps
        # L0 as pure forward-chain.
        # Iter 3 (2026-04-17): previous schedule still showed L3=0.90 spike
        # (vs L0=0.45). L3 at chain=5, contrapositive ON, adversarial ON
        # happened to be *easier* than L0 because at L0 the model still hit
        # the "Cannot be determined" option wrongly when rules were weak.
        # Delay contrapositive to L4+ so L3 remains pure forward-chain and
        # grow chain length more aggressively at L3+ to compensate.
        # Iter 4 (2026-04-17): L3=1.00 still spiking vs L0=0.40. Root cause:
        # at L0 with small chain (3), "Cannot be determined" trap catches
        # many seeds; at L3 with pure forward chain (no contrapositive) of
        # length 5 the conclusion is obvious. Fix: force L3 to use
        # contrapositive mode so at least 30% of rules are contrapositive
        # (harder to trace). Also reduce red-herring count boost.
        chain = min(8, 3 + level * 2 // 3)            # 3,3,4,5,5,6,7,7,8,8
        red = min(3, level // 3)                        # 0,0,0,1,1,1,2,2,2,3
        # Ensure chain+1+red <= 11.
        if chain + 1 + red > 11:
            red = 11 - chain - 1
        return {
            "chain_length":       chain,
            "n_red_herrings":     red,
            "use_contrapositive": level >= 3,          # was level >= 4
            "adversarial":        level >= 5,          # was level >= 3
            "biconditional":      level >= 6,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["chain_length"]

        for _ in range(30):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng: random.Random, cfg: Dict,
                      level: int) -> Optional[Tuple[str, str, Image.Image]]:
        n = cfg["chain_length"]
        n_rh = cfg["n_red_herrings"]
        use_contra = cfg["use_contrapositive"]
        adv = cfg["adversarial"]

        # Build a chain A → B → C → ... of n+1 propositions.
        if n + 1 + n_rh > len(_PROP_NAMES):
            return None
        names = rng.sample(_PROP_NAMES, n + 1 + n_rh)
        chain = names[: n + 1]
        red_herrings = names[n + 1:]

        rules: List[str] = []
        for i in range(n):
            if use_contra and rng.random() < 0.4:
                # Contrapositive version: "If not B then not A" instead of "If A then B".
                rules.append(
                    f"If {chain[i + 1]} is false, then {chain[i]} is false."
                )
            else:
                rules.append(
                    f"If {chain[i]} is true, then {chain[i + 1]} is true."
                )

        # Add red-herring rules.
        for rh in red_herrings:
            partner = rng.choice(names)
            if partner == rh:
                partner = rng.choice([x for x in names if x != rh])
            rules.append(
                f"If {rh} is true, then {partner} is true."
            )

        rng.shuffle(rules)

        # Starting fact: chain[0] is true.
        fact = f"{chain[0]} is true."
        # Correct conclusion: chain[n] is true.
        correct_stmt = f"{chain[n]} is true."

        # Distractors: other plausible conclusions.
        # A) {chain[n]} is false.
        # B) {chain[0]} is false.
        # C) "Cannot be determined"
        # Build a distractor that is NOT derivable from the chain. chain[1..n-1]
        # are all derivable via the rules, so using them would create multiple
        # correct answers. Use red_herrings when available; otherwise fall back
        # to a name that's NOT in the chain.
        all_names = list(_PROP_NAMES)
        chain_set = set(chain)
        if red_herrings:
            unused_dist = f"{rng.choice(red_herrings)} is true."
        else:
            outside = [n_ for n_ in all_names if n_ not in chain_set]
            if outside:
                unused_dist = f"{rng.choice(outside)} is true."
            else:
                unused_dist = f"{chain[0]} is false."  # safe fallback
        distractor_pool = [
            f"{chain[n]} is false.",
            f"{chain[0]} is false.",
            unused_dist,
            "Cannot be determined from the given rules.",
        ]
        if adv:
            # Add an "affirming the consequent" option.
            distractor_pool.append(
                f"If {chain[n]} is true, then {chain[0]} is true."
            )
        rng.shuffle(distractor_pool)
        distractors = distractor_pool[:3]
        options = [correct_stmt] + distractors
        rng.shuffle(options)
        if options.count(correct_stmt) > 1:
            return None
        answer_letter = chr(ord("A") + options.index(correct_stmt))

        # Include a short rule preview in the question so seed-diff test
        # catches level changes.
        n_rules_shown = len(rules)
        q_stems = [
            f"Read the {n_rules_shown} rules and the given fact ({fact}) shown in the image. Which conclusion must be TRUE?",
            f"Given {n_rules_shown} conditional rules and the starting fact ({fact}), which statement necessarily follows?",
            f"The image displays {n_rules_shown} if-then rules plus the fact: {fact}. Determine which conclusion is guaranteed.",
            f"Examine the {n_rules_shown} rules and initial fact ({fact}). Which option must hold?",
        ]
        question = (
            f"{rng.choice(q_stems)} Answer "
            f"with a single letter (A, B, C, or D)."
        )
        image = self._render(rules, fact, options, cfg)
        return question, answer_letter, image

    def _render(self, rules, fact, options, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        ff = self._rng.choice(["serif", "DejaVu Sans", "monospace"])
        fs = self._rng.choice([12, 13, 14])

        n_rules = len(rules)
        n_options = len(options)
        n_lines = n_rules + n_options + 7
        panel_h = max(8.0, n_lines * 0.58 + 1.0)

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

        logic_title_pool = ["Conditional Logic Chain", "If-Then Reasoning",
                            "Logical Deduction", "Rule Chain",
                            "Propositional Chain"]
        ax.text(5.0, panel_h - 0.05, self._rng.choice(logic_title_pool),
                fontsize=fs + 3, fontweight="bold",
                ha="center", va="top", family=ff, color="#2c3e50")
        y = panel_h - 0.85
        ax.text(0.6, y, "Rules:", fontsize=fs + 1, fontweight="bold",
                family=ff, color="#34495e", ha="left", va="top")
        y -= 0.55
        for i, r in enumerate(rules):
            ax.text(0.9, y, f"R{i + 1}. {r}", fontsize=fs, family=ff,
                    ha="left", va="top", color="#1a1a1a")
            y -= 0.55
        y -= 0.3
        ax.text(0.6, y, "Fact:", fontsize=fs + 1, fontweight="bold",
                family=ff, color="#34495e", ha="left", va="top")
        y -= 0.55
        ax.text(0.9, y, fact, fontsize=fs, family=ff,
                ha="left", va="top", color="#1a1a1a")
        y -= 0.75
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
    env = ConditionalLogicChainQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[seed={s} L{level}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"conditional_logic_chain_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[seed={s} L{level}] saved {path}")
            print(f"  Q (first 100): {env.get_instruction()[:100]}")
            print(f"  A: {env._answer}")
