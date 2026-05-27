"""
Latin Square Fill QA (D41).

Reference task:
  qid 319 (ES float): "In the grid, how many dark blue squares have to be
   coloured white, so that in each row and each column there is exactly one
   dark blue square? If it is impossible that in each row and each column
   there is exactly one dark blue square, answer 0." Ans: 10.

Render an N×N grid with some "dark blue" squares marked. The model must
determine how many marked squares need to be removed (recolored white) so
exactly one remains per row and per column. Answer 0 if impossible.

Verifier: integer answer.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to count the dark blue squares to remove from the {N}x{N} grid below.\n\n"
    "### Game Rules:\n"
    "1. Some cells of the {N}x{N} grid are colored dark blue; the rest are white.\n"
    "2. Recolor some dark blue squares to white so that each row and each column contains exactly one remaining dark blue square.\n"
    "3. Find the minimum number of dark blue squares that must be recolored white.\n"
    "4. If it is impossible to achieve exactly one dark blue square per row and per column, the answer is 0.\n\n"
    "### Coordinate System:\n"
    "- The grid is {N}x{N}, with rows top-to-bottom and columns left-to-right.\n"
    "- Each cell is either dark blue or white as shown in the image.\n\n"
    "### Current Puzzle State:\n"
    "- See the image: dark blue cells are filled, white cells are empty.\n\n"
    "### Output Format:\n"
    "Provide a single non-negative integer wrapped in <answer>...</answer>.\n"
    "Example: <answer>3</answer>",

    "Solve the dark-blue-removal puzzle on the {N}x{N} grid shown below.\n\n"
    "### Game Rules:\n"
    "- The grid contains some dark blue squares and some white squares.\n"
    "- Recolor (remove) some dark blue squares to white so that each row and each column has exactly one dark blue square left.\n"
    "- Report how many dark blue squares must be recolored.\n"
    "- If no recoloring can leave exactly one per row and per column, answer 0.\n\n"
    "### Coordinate System:\n"
    "- {N}x{N} grid; rows numbered 0..{N_minus_one} top-to-bottom, columns 0..{N_minus_one} left-to-right.\n\n"
    "### Current Puzzle State:\n"
    "- Refer to the image for the dark blue / white coloring of each cell.\n\n"
    "### Output Format:\n"
    "Output the integer count inside <answer>...</answer>.\n"
    "Example: <answer>0</answer>",

    "Determine the number of dark blue squares to remove on this {N}x{N} grid.\n\n"
    "### Game Rules:\n"
    "1. The grid is partly colored dark blue. Recolor some dark blue cells to white.\n"
    "2. After recoloring, each row must contain exactly one dark blue cell, and each column must contain exactly one dark blue cell.\n"
    "3. Output the minimum number of dark blue cells that need to be recolored.\n"
    "4. If no such recoloring exists (e.g. some row has no dark blue at all), output 0.\n\n"
    "### Coordinate System:\n"
    "- The grid has {N} rows and {N} columns, indexed (row, column) starting from 0.\n\n"
    "### Current Puzzle State:\n"
    "- The current dark blue / white pattern is shown in the image.\n\n"
    "### Output Format:\n"
    "Provide your final integer answer inside <answer>...</answer>.",
]


class LatinSquareFillQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "latin_square_fill"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 2:
            return {"N": 3}
        if level <= 5:
            return {"N": 4}
        if level <= 7:
            return {"N": 5}
        return {"N": 6}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        N = cfg["N"]
        rng = random.Random((self.seed or 0) * 5867 + level * 71 + 163)

        # 70% of seeds: have a valid completion (answer = #marks - N)
        # 30% of seeds: impossible (answer = 0)
        impossible = rng.random() < 0.3

        marks = set()  # (row, col)
        if impossible:
            # Force a row or column with 0 marks
            empty_row = rng.randint(0, N - 1)
            for r in range(N):
                if r == empty_row:
                    continue
                # Add a few marks in this row, none in the empty row
                ncells = rng.randint(1, N - 1)
                cols = rng.sample(range(N), ncells)
                for c in cols:
                    marks.add((r, c))
            # Make sure total marks > N for a tougher puzzle
            if len(marks) <= N:
                for _ in range(N):
                    r = rng.randint(0, N - 1)
                    c = rng.randint(0, N - 1)
                    if r != empty_row:
                        marks.add((r, c))
            answer_int = 0
        else:
            # Build a permutation, then add extra marks
            perm = list(range(N))
            rng.shuffle(perm)
            for r in range(N):
                marks.add((r, perm[r]))
            # Add extras
            n_extra = rng.randint(2, max(2, 2 * N))
            extras = 0
            attempts = 0
            while extras < n_extra and attempts < 100:
                attempts += 1
                r = rng.randint(0, N - 1)
                c = rng.randint(0, N - 1)
                if (r, c) not in marks:
                    marks.add((r, c))
                    extras += 1
            # Verify a valid permutation still exists (it does, since we put one
            # in). Also, no row should be empty (perm covers all).
            answer_int = len(marks) - N

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(N=N, N_minus_one=N - 1)
        answer = str(answer_int)
        img = self._render(N, marks)
        return question, answer, img

    def _render(self, N, marks) -> Image.Image:
        fig, ax = plt.subplots(figsize=(N + 0.5, N + 0.5), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for r in range(N):
            for c in range(N):
                color = "#1d4ed8" if (r, c) in marks else "#ffffff"
                ax.add_patch(plt.Rectangle((c, N - 1 - r), 1, 1,
                                           facecolor=color,
                                           edgecolor="#444",
                                           linewidth=1.2))
        ax.set_xlim(-0.1, N + 0.1)
        ax.set_ylim(-0.1, N + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = LatinSquareFillQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'NA'}")
            if ok:
                # The verifier picks the wrapper based on (seed % 3)
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                idx = (env.seed or 0) % 3
                v_ok = env.verify(wrappers[idx])["accuracy"]
                v_wrong = env.verify("definitely_wrong_xyz")["accuracy"]
                print(f"   right={v_ok} wrong={v_wrong}")
                if v_ok == 1 and v_wrong == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
