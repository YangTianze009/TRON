"""
Confusion-matrix diagonal-sum reading.

Render a K x K confusion-matrix heatmap with class labels on rows (true)
and columns (predicted), and integer counts annotated in each cell.
The model must read the diagonal cells (correct predictions) and report
their integer sum.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows a confusion matrix. Compute the sum of the diagonal entries (the count of correct predictions). Place the integer in <answer>...</answer>.",
    "Read the confusion matrix and add up all values on the main diagonal. Integer answer in <answer>...</answer>.",
    "What is the total number of correct predictions in the confusion matrix? (Sum of the main diagonal.) Integer in <answer>...</answer>.",
    "Add the diagonal entries of the confusion matrix shown. Place the integer total in <answer>...</answer>.",
    "Compute the sum of correctly classified instances by adding the entries on the matrix diagonal. Integer in <answer>...</answer>.",
    "From the confusion matrix in the image, compute the diagonal sum. Integer answer in <answer>...</answer>.",
    "Sum the values on the diagonal of the confusion matrix shown. Integer answer in <answer>...</answer>.",
    "Tally up the correctly-classified counts (entries on the matrix diagonal). Integer in <answer>...</answer>.",
    "The diagonal of the confusion matrix counts correct predictions. What is its total? Integer in <answer>...</answer>.",
    "Find the total of the diagonal cells in the confusion matrix. Integer answer in <answer>...</answer>.",
    "How many predictions were correct overall? (Add the diagonal of the confusion matrix.) Integer in <answer>...</answer>.",
    "Add together every cell on the main diagonal of the confusion matrix shown. Integer answer in <answer>...</answer>.",
    "Calculate the total number of correctly classified samples from the confusion matrix. Integer in <answer>...</answer>.",
    "Read the matrix and report the sum of its diagonal. Integer in <answer>...</answer>.",
    "What integer do you get when summing every cell on the diagonal of the confusion matrix shown? Place it in <answer>...</answer>.",
    "Compute the sum across the matrix's main diagonal (top-left to bottom-right). Integer in <answer>...</answer>.",
    "Inspect the confusion matrix and report the integer sum of its diagonal entries. Place the answer in <answer>...</answer>.",
    "Read the diagonal cells of the matrix and add them up. Integer answer in <answer>...</answer>.",
]


_CLASS_LABEL_POOLS = [
    ["Cat", "Dog", "Bird", "Fish", "Horse", "Cow", "Sheep", "Goat"],
    ["A", "B", "C", "D", "E", "F", "G", "H"],
    ["Red", "Green", "Blue", "Yellow", "Purple", "Orange", "Black", "White"],
    ["Setosa", "Virginica", "Versicolor", "Other1", "Other2", "Other3", "Other4", "Other5"],
    ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7", "Class8"],
    ["Spam", "Ham", "Promo", "Update", "Forum", "Social", "Other", "Junk"],
    ["Buy", "Sell", "Hold", "Skip", "Watch", "Trade", "Hedge", "Exit"],
]


class ConfusionMatrixDiagonalQA(StandaloneVisualEnv):
    ENV_NAME = "confusion_matrix_diagonal"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # K x K. Trivial L0 = 2x2.
        if level == 0:
            return {"K": 2, "v_lo": 1, "v_hi": 9, "trivial_l0": True}
        if level <= 2:
            return {"K": 3, "v_lo": 1, "v_hi": 12, "trivial_l0": False}
        if level <= 4:
            return {"K": 4, "v_lo": 1, "v_hi": 18, "trivial_l0": False}
        if level <= 6:
            return {"K": 5, "v_lo": 1, "v_hi": 25, "trivial_l0": False}
        if level <= 8:
            return {"K": 6, "v_lo": 0, "v_hi": 30, "trivial_l0": False}
        return {"K": 7, "v_lo": 0, "v_hi": 35, "trivial_l0": False}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1117 + level * 97 + 41)

        K = cfg["K"]
        labels = rng.sample(_CLASS_LABEL_POOLS[rng.randrange(
            len(_CLASS_LABEL_POOLS))], K)

        if cfg["trivial_l0"]:
            # 2x2 matrix [[5,1],[2,7]]; diag sum = 12
            mat = np.array([[5, 1], [2, 7]], dtype=int)
        else:
            mat = np.zeros((K, K), dtype=int)
            for i in range(K):
                for j in range(K):
                    if i == j:
                        # diagonal usually larger to look like a real
                        # classifier, but we don't depend on this.
                        mat[i, j] = rng.randint(
                            cfg["v_lo"] + (cfg["v_hi"] - cfg["v_lo"]) // 2,
                            cfg["v_hi"])
                    else:
                        mat[i, j] = rng.randint(cfg["v_lo"], cfg["v_hi"] // 2)

        diag_sum = int(np.trace(mat))

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer = str(diag_sum)
        img = self._render(mat, labels, rng)
        return question, answer, img

    def _render(self, mat: np.ndarray, labels: List[str],
                rng: random.Random) -> Image.Image:
        K = mat.shape[0]
        fig_w = max(5.5, 1.0 * K + 2.5)
        fig_h = max(4.8, 1.0 * K + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        cmap_choice = rng.choice(["Blues", "Greens", "Purples", "Oranges",
                                   "viridis", "YlGnBu"])
        im = ax.imshow(mat, cmap=cmap_choice, aspect="equal")

        # Annotate cells with integer counts. Choose text color based on
        # cell intensity so text stays readable on any cmap.
        vmax = mat.max() if mat.max() > 0 else 1
        for i in range(K):
            for j in range(K):
                v = int(mat[i, j])
                # Threshold for white-on-dark vs black-on-light.
                color = "white" if mat[i, j] > 0.55 * vmax else "#111111"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=12, color=color, fontweight="bold")

        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=12)

        # Colorbar adds heatmap-style polish.
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
