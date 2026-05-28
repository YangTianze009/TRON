"""
Set Operation on Lists QA (D1).

Reference task:
  an external reference (HS MCQ): "What is the intersection of sets A and B?
   A: [1, 4, 13]; B: [3, 7, 12, 15, 18]; C: [3, 5, 7, 10, 12, 15, 18];
   D: [5, 10]." Ans: A.
  an external reference (HS MCQ): "What is the union of sets A and B? ..." Ans: D.
  an external reference (HS MCQ): "Which elements are unique to set A? ..." Ans: B.
  an external reference (HS MCQ): "Which elements are unique to set B? ..." Ans: C.

Renders two listed sets A and B (as image of "A = {...}, B = {...}"),
asks for one of {union, intersection, set-diff A\\B, set-diff B\\A}, with
4 list options as MCQ.

Verifier: single MCQ letter A-D.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_OP_TEMPLATES = {
    "intersection": [
        "What is the intersection of sets A and B (A ∩ B)? Choose the matching list. Place letter in <answer>...</answer>.",
        "Determine A ∩ B from the listed sets. Place the letter of the matching option in <answer>...</answer>.",
        "Pick the option that lists exactly the elements common to both sets A and B. Letter in <answer>...</answer>.",
    ],
    "union": [
        "What is the union of sets A and B (A ∪ B)? Choose the matching list. Place letter in <answer>...</answer>.",
        "Determine A ∪ B from the listed sets. Place the letter of the matching option in <answer>...</answer>.",
        "Pick the option that lists every element appearing in A or B (no duplicates). Letter in <answer>...</answer>.",
    ],
    "diff_a_b": [
        "Which elements are unique to set A (in A but not in B)? Place letter in <answer>...</answer>.",
        "Determine A \\\\ B (elements of A not in B). Choose the matching list. Letter in <answer>...</answer>.",
        "Pick the list of elements that belong to A only (not in B). Letter in <answer>...</answer>.",
    ],
    "diff_b_a": [
        "Which elements are unique to set B (in B but not in A)? Place letter in <answer>...</answer>.",
        "Determine B \\\\ A (elements of B not in A). Choose the matching list. Letter in <answer>...</answer>.",
        "Pick the list of elements that belong to B only (not in A). Letter in <answer>...</answer>.",
    ],
}


def _set_to_list_str(s):
    return "[" + ", ".join(str(x) for x in sorted(s)) + "]"


class SetOperationListsQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "set_operation_lists"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # element pool size and set size grow with level
        if level <= 2:
            return {"pool_max": 12, "set_size": 3}
        if level <= 4:
            return {"pool_max": 18, "set_size": 4}
        if level <= 6:
            return {"pool_max": 22, "set_size": 5}
        if level <= 8:
            return {"pool_max": 25, "set_size": 6}
        # 2026-05-04: bumped L9 difficulty (re-bump) — still saturated at 95%.
        # Even bigger pool + larger sets + nested op flag.
        return {"pool_max": 80, "set_size": 14, "nested_op": True}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7919 + level * 71 + 13)

        op = rng.choice(["intersection", "union", "diff_a_b", "diff_b_a"])
        # Generate sets A and B with a forced overlap so all ops are interesting
        for _ in range(50):
            pool = list(range(1, cfg["pool_max"] + 1))
            rng.shuffle(pool)
            n_overlap = rng.randint(1, max(1, cfg["set_size"] - 1))
            common = pool[:n_overlap]
            rest = pool[n_overlap:]
            unique_a = rest[:cfg["set_size"] - n_overlap]
            unique_b = rest[cfg["set_size"] - n_overlap:
                            cfg["set_size"] - n_overlap +
                            (cfg["set_size"] - n_overlap)]
            A = set(common + unique_a)
            B = set(common + unique_b)
            if len(A) == cfg["set_size"] and len(B) == cfg["set_size"]:
                break
        else:
            return None

        if op == "intersection":
            correct_set = A & B
        elif op == "union":
            correct_set = A | B
        elif op == "diff_a_b":
            correct_set = A - B
        else:
            correct_set = B - A

        # Generate distractors: 3 wrong sets that are also valid set-op
        # results but differ from correct (so model can't shortcut by ruling
        # out impossible).
        distractors = set()
        all_options = [
            (A & B), (A | B), (A - B), (B - A),
        ]
        for o in all_options:
            if o != correct_set:
                distractors.add(frozenset(o))
        # If we don't have enough yet, perturb
        attempts = 0
        while len(distractors) < 3 and attempts < 30:
            attempts += 1
            base = list(correct_set) if correct_set else [1]
            # perturb
            extra_pool = [x for x in range(1, cfg["pool_max"] + 1)
                          if x not in correct_set]
            rng.shuffle(extra_pool)
            n_swap = rng.randint(1, 2)
            new_set = set(base)
            for k in range(min(n_swap, len(base))):
                new_set.discard(base[k])
            for k in range(n_swap):
                if k < len(extra_pool):
                    new_set.add(extra_pool[k])
            if new_set != correct_set and frozenset(new_set) not in distractors:
                distractors.add(frozenset(new_set))
        # Build options. Ground truth at random position.
        opts = list(distractors)[:3]
        opts = [set(o) for o in opts]
        correct_pos = rng.randint(0, 3)
        full_opts = []
        for i in range(4):
            if i == correct_pos:
                full_opts.append(correct_set)
            else:
                if opts:
                    full_opts.append(opts.pop())
                else:
                    full_opts.append({-1})  # fallback
        answer_letter = "ABCD"[correct_pos]

        question = rng.choice(_OP_TEMPLATES[op])
        img = self._render(A, B, full_opts, op)
        return question, answer_letter, img

    def _render(self, A, B, options, op) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        op_label = {
            "intersection": "A ∩ B = ?",
            "union": "A ∪ B = ?",
            "diff_a_b": "A \\ B = ?",
            "diff_b_a": "B \\ A = ?",
        }[op]
        # Top: sets
        ax.text(0.5, 9.0, "A = " + _set_to_list_str(A), fontsize=15,
                fontweight="bold", color="#003566", va="center")
        ax.text(0.5, 8.0, "B = " + _set_to_list_str(B), fontsize=15,
                fontweight="bold", color="#7f1d1d", va="center")
        ax.text(0.5, 6.8, "Question: " + op_label, fontsize=15,
                fontweight="bold", color="#1a1a1a")
        # Options
        for i, opt in enumerate(options):
            letter = "ABCD"[i]
            y = 5.5 - i * 1.1
            ax.text(0.5, y,
                    f"({letter}) " + _set_to_list_str(opt),
                    fontsize=14, color="#1a1a1a", va="center")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = SetOperationListsQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
