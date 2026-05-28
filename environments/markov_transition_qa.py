"""
Markov Chain Transition Probability QA (D117, P3 — reference statistics).

Reference an external reference:
  "According to the markov chain shown in the image, what is the
   probability of the event 'A to B'?"  Ans: 0.1

This env renders a Markov-chain state diagram (3-5 nodes) with directed
edges labeled by transition probabilities. Question types:
  - 'one_step'   — P(A -> B) given the diagram (read off direct edge)
  - 'two_step'   — P(A -> ? -> B) summed over intermediate states
  - 'steady_state' — for a 2-state chain, compute the long-run P(B)

Verifier: float answer (`\\boxed{0.1}` or bare 0.1).

Difficulty:
  L0..L2 — 3 states, one-step probability.
  L3..L5 — 4 states, one-step or two-step.
  L6..L7 — 5 states, two-step.
  L8..L9 — 2-state steady state, or two-step on 4 states.
"""
import math
import random
import string
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


def _normalize(weights: List[int]) -> List[float]:
    s = sum(weights)
    if s == 0:
        return [0.0] * len(weights)
    return [w / s for w in weights]


def _round_clean(p: float) -> float:
    """Round to a 'nice' value: 0.1, 0.2, ..., 0.5, etc."""
    return round(p, 2)


class MarkovTransitionQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "markov_transition"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            return {"n_states": 3, "qtype": "one_step"}
        if level <= 4:
            return {"n_states": 4, "qtype": "one_step"}
        if level <= 5:
            return {"n_states": 4, "qtype": "two_step"}
        if level <= 7:
            return {"n_states": 5, "qtype": "two_step"}
        if level <= 8:
            return {"n_states": 2, "qtype": "steady_state"}
        return {"n_states": 4, "qtype": "two_step"}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 9001 + level * 173 + 41)

        for _ in range(20):
            r = self._try_generate(rng, cfg)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg):
        n = cfg["n_states"]
        qtype = cfg["qtype"]
        labels = list(string.ascii_uppercase[:n])  # A, B, C, ...

        # Build transition matrix with nice tenths-place probabilities so
        # answers stay clean.
        # For each row, choose 2-4 outgoing edges, each weight ∈ {1..9}.
        T = [[0.0] * n for _ in range(n)]
        edges_with_weight = []  # for rendering: (i, j, prob)
        for i in range(n):
            n_out = min(n, rng.randint(2, min(4, n)))
            outgoing = rng.sample(range(n), n_out)
            # Ensure self-loops are possible
            # Use integer weights summing to 10 → tenths-place probabilities.
            # Sample n_out positive integers summing to 10.
            wts = self._random_partition(10, n_out, rng)
            if any(w == 0 for w in wts):
                return None
            for k, j in enumerate(outgoing):
                p = wts[k] / 10.0
                T[i][j] = p
                edges_with_weight.append((i, j, p))

        # Pick question
        if qtype == "one_step":
            # Pick start with at least one outgoing edge
            valid_starts = [i for i in range(n) if any(T[i][j] > 0 for j in range(n))]
            if not valid_starts:
                return None
            i = rng.choice(valid_starts)
            # Pick a target with non-zero P
            targets_pos = [j for j in range(n) if T[i][j] > 0]
            if not targets_pos:
                return None
            # Mix in some asks for zero-probability transitions (~25%)
            if rng.random() < 0.25 and len(targets_pos) < n:
                j_choices = [j for j in range(n) if T[i][j] == 0]
                if j_choices:
                    j = rng.choice(j_choices)
                    answer = 0.0
                else:
                    j = rng.choice(targets_pos)
                    answer = T[i][j]
            else:
                j = rng.choice(targets_pos)
                answer = T[i][j]
            question_core = (
                f"According to the Markov chain shown in the image, what "
                f"is the probability of the event '{labels[i]} to "
                f"{labels[j]}' (one transition step)?"
            )
        elif qtype == "two_step":
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                # Pick distinct
                j = (j + rng.randint(1, n - 1)) % n
            # Compute P(i->j) in two steps = sum_k T[i][k]*T[k][j]
            answer = sum(T[i][k] * T[k][j] for k in range(n))
            answer = round(answer, 4)
            if answer <= 0.0001 or answer >= 1.0:
                return None
            question_core = (
                f"According to the Markov chain shown in the image, what "
                f"is the probability of being in state {labels[j]} after "
                f"exactly 2 transitions starting from state {labels[i]}?"
            )
        else:  # steady_state
            # 2-state chain. Steady state for state 1 (B):
            # pi_B = P(A->B) / (P(A->B) + P(B->A))
            p_AB = T[0][1]
            p_BA = T[1][0]
            if p_AB + p_BA == 0:
                return None
            answer = p_AB / (p_AB + p_BA)
            answer = round(answer, 4)
            question_core = (
                f"According to the 2-state Markov chain shown in the "
                f"image, what is the long-run (steady-state) probability "
                f"of being in state {labels[1]}?"
            )

        # Format answer string
        # Round to 2 or 4 decimals depending on cleanness
        if abs(answer - round(answer, 2)) < 1e-6:
            ans_str = f"{round(answer, 2):g}"
        else:
            ans_str = f"{answer:.4f}"

        # Render
        img = self._render(n, labels, edges_with_weight)
        question = question_core + "\n\nProvide your answer as a decimal number."
        return question, ans_str, img

    @staticmethod
    def _random_partition(total: int, k: int, rng: random.Random) -> List[int]:
        """Random positive integers summing to `total` with `k` parts."""
        if k > total:
            return [0] * k
        # Stars and bars: pick k-1 distinct cuts from {1..total-1}
        cuts = sorted(rng.sample(range(1, total), k - 1)) if k > 1 else []
        cuts = [0] + cuts + [total]
        return [cuts[i + 1] - cuts[i] for i in range(k)]

    # ------------------------------------------------------------------ #
    def _render(self, n: int, labels: List[str], edges) -> Image.Image:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.0), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Place nodes on a circle
        positions = {}
        radius = max(2.5, n * 0.55)
        for i in range(n):
            angle = 2 * math.pi * i / n - math.pi / 2
            positions[i] = (radius * math.cos(angle), radius * math.sin(angle))

        # Draw edges
        edge_set = set((s, t) for s, t, _w in edges)
        for s, t, p in edges:
            sx, sy = positions[s]
            tx, ty = positions[t]
            if s == t:
                # Self-loop drawn as small arc tangent to node circle.
                # Direction: outward from centre (origin) along the
                # node's radial direction.
                d_norm = math.hypot(sx, sy) + 1e-9
                dx, dy = sx / d_norm, sy / d_norm
                # Loop centre = node + outward offset
                loop_cx = sx + dx * 0.65
                loop_cy = sy + dy * 0.65
                circ = plt.Circle((loop_cx, loop_cy), 0.32,
                                  fill=False, edgecolor="#7f8c8d",
                                  linewidth=1.4)
                ax.add_patch(circ)
                # Small arrow indicating direction
                ax.annotate(
                    "", xy=(sx + dx * 0.42, sy + dy * 0.42),
                    xytext=(loop_cx + dx * 0.05, loop_cy + dy * 0.05),
                    arrowprops=dict(arrowstyle="-|>", color="#7f8c8d",
                                    lw=1.2),
                )
                ax.text(loop_cx + dx * 0.5, loop_cy + dy * 0.5,
                        f"{p:g}",
                        fontsize=10, fontweight="bold", color="#e74c3c",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.85))
                continue

            has_reverse = (t, s) in edge_set
            rad = 0.18 if has_reverse else 0.0
            ax.annotate(
                "", xy=(tx, ty), xytext=(sx, sy),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#7f8c8d",
                    lw=1.5,
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=20, shrinkB=20,
                ),
            )
            mx = (sx + tx) / 2
            my = (sy + ty) / 2
            if has_reverse:
                dx = tx - sx
                dy = ty - sy
                length = math.hypot(dx, dy) + 1e-9
                nx_off = -dy / length * 0.42
                ny_off = dx / length * 0.42
                mx += nx_off
                my += ny_off
            ax.text(mx, my, f"{p:g}", fontsize=11, fontweight="bold",
                    color="#e74c3c", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="none", alpha=0.85))

        # Draw nodes
        for i in range(n):
            x, y = positions[i]
            color = "#3498db"
            circle = plt.Circle((x, y), 0.42, facecolor=color,
                                edgecolor="#2c3e50", linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, labels[i], fontsize=15, fontweight="bold",
                    color="white", ha="center", va="center", zorder=6)

        margin = max(3.5, radius + 1.4)
        ax.set_xlim(-margin, margin)
        ax.set_ylim(-margin, margin)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Markov chain", fontsize=13, fontweight="bold")
        return self.fig_to_pil(fig, dpi=110)


if __name__ == "__main__":
    env = MarkovTransitionQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6, 9):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok}; A={env._answer if ok else 'X'}")
            if ok:
                v = env.verify(f"\\boxed{{{env._answer}}}")
                v2 = env.verify(f"<answer>{env._answer}</answer>")
                v3 = env.verify("definitely_wrong_xyz")
                print(f"   boxed={v['accuracy']} answer={v2['accuracy']} wrong={v3['accuracy']}")
                if v['accuracy'] == 1 and v2['accuracy'] == 1 and v3['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
