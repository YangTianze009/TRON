"""
Gantt chart QA — timeline reading, overlap detection, total duration.
Targets: chart-reading, a multimodal benchmark logical reasoning, diagram process diagrams.
Capabilities: V3+V9 (chart + flow), R1 (arithmetic), R5 (multi-step)
"""
import random, math
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class GanttChartQA(StandaloneVisualEnv):
    ENV_NAME = "gantt_chart"

    _TASK_POOLS = [
        ["Design", "Develop", "Test", "Deploy", "Review", "Document"],
        ["Planning", "Research", "Prototype", "Build", "QA", "Launch"],
        ["Analysis", "Coding", "Testing", "Integration", "Release"],
    ]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_tasks": (4, 4), "qtypes": ["task_duration"]}
        if level == 1:
            return {"n_tasks": (4, 5), "qtypes": ["task_duration", "which_longest"]}
        if level == 2:
            return {"n_tasks": (4, 5), "qtypes": ["which_longest", "task_end_time"]}
        if level == 3:
            return {"n_tasks": (4, 5), "qtypes": ["task_end_time", "total_project_duration"]}
        if level == 4:
            return {"n_tasks": (5, 6), "qtypes": ["total_project_duration", "overlap_count"]}
        if level == 5:
            return {"n_tasks": (5, 6), "qtypes": ["overlap_count", "total_project_duration"]}
        if level == 6:
            return {"n_tasks": (5, 6), "qtypes": ["overlap_count", "resource_conflict_count"]}
        if level == 7:
            return {"n_tasks": (5, 6), "qtypes": ["resource_conflict_count", "slack_time"]}
        if level == 8:
            return {"n_tasks": (5, 6), "qtypes": ["slack_time"]}
        return {"n_tasks": (5, 6), "qtypes": ["slack_time", "resource_conflict_count"]}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 715)
        question_type = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        pool = sub_rng.choice(self._TASK_POOLS)
        lo_n, hi_n = cfg["n_tasks"]
        n = sub_rng.randint(lo_n, min(hi_n, len(pool)))
        tasks = pool[:n]

        # Generate start/duration for each task
        starts = []
        durations = []
        t = 0
        for i in range(n):
            gap = rng.randint(0, 3)
            overlap = rng.randint(0, 2)
            start = max(0, t - overlap + gap)
            dur = rng.randint(2, 8)
            starts.append(start)
            durations.append(dur)
            t = start + dur

        ends = [s + d for s, d in zip(starts, durations)]
        total_end = max(ends)

        # Q&A
        if question_type == "task_duration":
            idx = rng.randint(0, n-1)
            question = f"How long does '{tasks[idx]}' take (in time units)?"
            answer = durations[idx]
        elif question_type == "total_project_duration":
            question = "What is the total project duration (from start of first task to end of last)?"
            answer = total_end - min(starts)
        elif question_type == "which_longest":
            idx = durations.index(max(durations))
            question = "Which task has the longest duration?"
            answer = tasks[idx]
        elif question_type == "overlap_count":
            overlaps = 0
            for i in range(n):
                for j in range(i+1, n):
                    if starts[i] < ends[j] and starts[j] < ends[i]:
                        overlaps += 1
            question = "How many pairs of tasks overlap in time?"
            answer = overlaps
        elif question_type == "task_end_time":
            idx = rng.randint(0, n-1)
            question = f"At what time does '{tasks[idx]}' end?"
            answer = ends[idx]
        elif question_type == "slack_time":
            # Slack = project_end - task_end - sum of durations of tasks that depend on it
            # Simplified: slack = (total_end - ends[i]) for each task
            # Find the non-critical task with the most slack
            project_end = total_end - min(starts)
            slacks = [(total_end - ends[i], tasks[i]) for i in range(n)]
            # Filter to non-zero slack (non-critical tasks)
            non_critical = [(s, t) for s, t in slacks if s > 0]
            if not non_critical:
                return None
            max_slack = max(non_critical, key=lambda x: x[0])
            question = (f"The total project runs from time {min(starts)} to {total_end}. "
                        f"Slack time for a task = project end time - task end time. "
                        f"Which non-critical task (slack > 0) has the most slack time, "
                        f"and what is that slack? Answer as 'TaskName, slack'.")
            answer = f"{max_slack[1]}, {max_slack[0]}"
        elif question_type == "resource_conflict_count":
            # Count the maximum number of tasks running at any single time point
            max_concurrent = 0
            for t in range(min(starts), total_end + 1):
                count = sum(1 for i in range(n) if starts[i] <= t < ends[i])
                max_concurrent = max(max_concurrent, count)
            question = ("What is the maximum number of tasks running concurrently "
                        "at any single point in time?")
            answer = max_concurrent
        else:
            return None

        # Render
        style = self._random_style()
        fig, ax = plt.subplots(figsize=(max(8, total_end * 0.8), max(3, n * 0.7 + 1)))
        self._apply_style(fig, ax, style)
        palette = style["palette"]

        for i in range(n):
            color = palette[i % len(palette)]
            ax.barh(i, durations[i], left=starts[i], height=0.6, color=color,
                   edgecolor="black", linewidth=1, alpha=0.8)
            ax.text(starts[i] + durations[i]/2, i, f"{durations[i]}",
                   ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        ax.set_yticks(range(n))
        ax.set_yticklabels(tasks, fontsize=style["font_size_base"])
        ax.set_xlabel("Time", fontsize=style["font_size_base"])
        ax.set_xlim(-0.5, total_end + 1)
        ax.invert_yaxis()
        ax.set_title("Gantt Chart — Project Timeline", fontsize=style["font_size_base"] + 2, fontweight="bold")

        return question, str(answer), self.fig_to_pil(fig, dpi=style["dpi"])
