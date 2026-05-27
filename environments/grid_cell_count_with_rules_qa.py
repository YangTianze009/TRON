"""
Grid Cell Count With Rules QA.

NxN grid where each cell has a color and/or number. A rule is stated and the
model must count cells satisfying the rule.

Difficulty axes:
  A) grid_size: 3..7
  B) rule_complexity: simple threshold -> neighbor-dependent
"""
import random
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from .standalone_base import StandaloneVisualEnv

_CELL_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']

class GridCellCountWithRulesQA(StandaloneVisualEnv):
    ENV_NAME = "grid_cell_count_with_rules"

    def _level_config(self, level):
        return {
            'grid_size': 3 + level // 2,
            'rule_type': (['threshold'] if level <= 2
                          else ['threshold', 'neighbor'] if level <= 5
                          else ['threshold', 'neighbor', 'conjunction']),
        }

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 1012)
        style = self._random_style()

        n = cfg['grid_size']
        rule_type = rng.choice(cfg['rule_type'])

        # Generate grid values and colors
        values = [[rng.randint(1, 9) for _ in range(n)] for _ in range(n)]
        n_colors = min(3 + level // 3, len(_CELL_COLORS))
        colors = [[rng.choice(_CELL_COLORS[:n_colors]) for _ in range(n)] for _ in range(n)]

        # Define rule and count
        if rule_type == 'threshold':
            thresh = rng.randint(4, 7)
            rule_text = f"Count cells where the number is greater than {thresh}."
            count = sum(1 for r in range(n) for c in range(n) if values[r][c] > thresh)
        elif rule_type == 'neighbor':
            rule_text = "Count cells where the number is greater than the number of adjacent cells with the same color."
            count = 0
            for r in range(n):
                for c in range(n):
                    same_neighbors = 0
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc2 = r+dr, c+dc
                        if 0 <= nr < n and 0 <= nc2 < n and colors[nr][nc2] == colors[r][c]:
                            same_neighbors += 1
                    if values[r][c] > same_neighbors:
                        count += 1
        else:  # conjunction
            thresh = rng.randint(3, 6)
            target_color = rng.choice(_CELL_COLORS[:n_colors])
            cname = {'#e74c3c':'red','#3498db':'blue','#2ecc71':'green',
                     '#f1c40f':'yellow','#9b59b6':'purple','#e67e22':'orange'}.get(target_color, 'colored')
            rule_text = f"Count cells where the number is at least {thresh} AND the cell is {cname}."
            count = sum(1 for r in range(n) for c in range(n)
                        if values[r][c] >= thresh and colors[r][c] == target_color)

        answer = str(count)

        # Draw grid
        sc = style['figsize_scale']
        fig, ax = plt.subplots(figsize=(max(5, n+1)*sc, max(5, n+1)*sc))
        fig.patch.set_facecolor(style['bg_color'])
        ax.set_facecolor(style['bg_color'])
        ax.set_xlim(-0.1, n+0.1); ax.set_ylim(-0.1, n+0.1)
        ax.set_aspect('equal'); ax.axis('off')

        for r in range(n):
            for c in range(n):
                # Plain Rectangle: FancyBboxPatch's default boxstyle pad was
                # extending cells beyond their grid bounds and creating an
                # apparent "double-layer" overlay that confused the model.
                rect = mpatches.Rectangle((c, n-1-r), 1, 1,
                    facecolor=colors[r][c], edgecolor='#333', linewidth=1.5, alpha=0.7)
                ax.add_patch(rect)
                ax.text(c+0.5, n-0.5-r, str(values[r][c]),
                       ha='center', va='center', fontsize=style['font_size_base']+2,
                       fontweight='bold', color='#111')

        ax.set_title("Grid Cell Count", fontsize=style['font_size_base']+3, fontweight='bold')
        try: fig.tight_layout()
        except: pass
        img = self.fig_to_pil(fig, dpi=style['dpi'])

        q = f"Rule: {rule_text} Answer with a single integer."
        return q, answer, img

if __name__ == "__main__":
    env = GridCellCountWithRulesQA()
    for lv in [0, 6]:
        ok = env.generate(42, {"level": lv})
        print(f"L{lv}: ok={ok}, answer={env._answer}")
