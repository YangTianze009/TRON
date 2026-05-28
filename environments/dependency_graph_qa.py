"""
Dependency/task graph QA — critical path, parallel execution, scheduling.
Targets: diagram (flow diagrams), a multimodal benchmark logical reasoning.

Capabilities: V9 (arrow/flow parsing), R1 (arithmetic), R5 (multi-step)
"""
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv
from ._render_modes import pick_render_mode, textbook_params, sketch_context

_Q_SEQ_TIME = [
    "If all tasks run sequentially (one after another), what is the total time?",
    "Assuming strict sequential execution of the tasks, what total time is required?",
    "If the tasks must run one at a time in sequence, how long does the entire schedule take?",
    "Running every task back-to-back (no parallelism), what is the combined duration?",
    "What is the sum of durations if we execute all tasks sequentially?",
    "Executing all tasks one after another, how many time units are needed in total?",
    "Compute the total elapsed time when running the tasks strictly in sequence.",
    "If only one task can run at a time, what is the minimum total schedule length?",
    "Total time for purely sequential execution of all tasks — report the number.",
    "How many time units are needed if the tasks are done one at a time?",
    "Sum up the task durations (purely sequential execution). What is the total?",
    "Running all tasks strictly sequentially, what duration does the schedule have?",
    "Calculate the total schedule length under sequential (no-parallel) execution.",
    "In a schedule where tasks run one after another, how long is the total project?",
    "What total time results from running every task strictly sequentially?",
    "Assuming single-threaded execution (tasks in sequence), what is the overall duration?",
]

_Q_CRIT_PATH = [
    "What is the minimum total time to complete all tasks (critical path length)?",
    "What is the length of the critical path (i.e., the minimum makespan)?",
    "Assuming unlimited parallelism, what is the earliest all tasks can finish?",
    "Compute the critical path length of the project — the minimum completion time.",
    "With full parallelism allowed, what is the shortest total time to finish every task?",
    "What is the minimum possible project duration respecting all dependencies?",
    "Determine the critical-path time — the fastest way to finish all tasks.",
    "What is the makespan (critical path length) of this task graph?",
    "Assuming enough workers, what's the shortest possible project completion time?",
    "What is the length of the longest dependency chain through this graph (critical path)?",
    "Given the dependencies, how soon can all tasks be complete under unlimited concurrency?",
    "Compute the minimum finish time for the entire project.",
    "Under optimal parallel scheduling, what total time does the project take?",
    "What is the earliest time that all tasks can be finished, given the dependencies?",
    "Find the critical path length for this dependency graph.",
    "Report the minimum possible makespan for this schedule.",
]

_Q_PARALLEL = [
    "Which tasks can run in parallel at time 0 (no dependencies)? List the count.",
    "How many tasks have no prerequisites (can start immediately)? Give the count.",
    "Count the number of tasks that can begin at time 0.",
    "How many tasks have an empty dependency set in this graph?",
    "Report the number of tasks that can launch simultaneously at t=0.",
    "How many tasks are ready to run at time 0 (no incoming dependencies)?",
    "Count the tasks with no predecessors.",
    "At time zero, how many tasks are eligible to start? Give the integer.",
    "What is the count of starter tasks (tasks with no dependencies)?",
    "How many nodes in the dependency graph have in-degree zero?",
    "Report how many tasks could all be started at once at the beginning.",
    "How many tasks can run concurrently at t=0 (no deps)? Integer answer.",
    "Count tasks that do not depend on any other task.",
    "Give the count of initially-ready tasks (no prerequisites).",
    "How many tasks can launch in parallel at the start of the schedule?",
    "Report the number of tasks with zero prerequisites.",
]

_Q_EARLIEST = [
    "What is the earliest possible start time for '{target}'?",
    "When is the earliest time '{target}' can begin, given the dependencies and durations?",
    "Compute the earliest start time of '{target}' in the schedule.",
    "What is the minimum start time of task '{target}'?",
    "At what time (earliest) can task '{target}' begin?",
    "Determine when '{target}' can first start, respecting its dependencies.",
    "What is the earliest start of '{target}' under optimal scheduling?",
    "Report the earliest possible start time for task '{target}'.",
    "Given the graph, when can '{target}' at earliest begin?",
    "What is the value of ES('{target}') — the earliest start time?",
    "When is the soonest that '{target}' can start?",
    "For task '{target}', compute the earliest-start time.",
    "Given the dependencies and durations, at what earliest time does '{target}' become ready?",
    "Find the earliest start time of '{target}'.",
    "At minimum, by what time is task '{target}' ready to begin?",
    "State the earliest time task '{target}' can start.",
]

class DependencyGraphQA(StandaloneVisualEnv):
    ENV_NAME = "dependency_graph"

    _QTYPES = ["total_sequential_time", "critical_path_time",
               "which_can_parallel", "earliest_start"]

    def _level_config(self, level: int) -> Dict:
        if level <= 0:
            return {"n_tasks_range": (4, 4), "dur_hi": 5, "qtypes": ["total_sequential_time"]}
        if level == 1:
            return {"n_tasks_range": (4, 5), "dur_hi": 6, "qtypes": ["total_sequential_time", "which_can_parallel"]}
        if level == 2:
            return {"n_tasks_range": (4, 5), "dur_hi": 6, "qtypes": ["which_can_parallel", "total_sequential_time"]}
        if level == 3:
            return {"n_tasks_range": (5, 6), "dur_hi": 7, "qtypes": ["critical_path_time", "which_can_parallel"]}
        if level == 4:
            return {"n_tasks_range": (5, 6), "dur_hi": 7, "qtypes": ["critical_path_time", "earliest_start"]}
        if level == 5:
            return {"n_tasks_range": (5, 7), "dur_hi": 8, "qtypes": ["earliest_start", "critical_path_time"]}
        if level == 6:
            return {"n_tasks_range": (5, 7), "dur_hi": 8, "qtypes": ["earliest_start", "critical_path_time"]}
        if level == 7:
            return {"n_tasks_range": (6, 7), "dur_hi": 8, "qtypes": ["earliest_start", "critical_path_time"]}
        if level == 8:
            return {"n_tasks_range": (6, 7), "dur_hi": 8, "qtypes": ["earliest_start"]}
        return {"n_tasks_range": (6, 7), "dur_hi": 8, "qtypes": ["earliest_start", "critical_path_time"]}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        rng = self._rng
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        sub_rng = random.Random((seed or 0) * 1000 + level * 37 + 703)
        question_type = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        # Generate tasks
        lo, hi = cfg["n_tasks_range"]
        n_tasks = sub_rng.randint(lo, hi)
        tasks = [f"T{i+1}" for i in range(n_tasks)]
        durations = {t: sub_rng.randint(1, cfg.get("dur_hi", 8)) for t in tasks}

        # Generate dependencies (DAG)
        deps = {t: [] for t in tasks}
        for i in range(1, n_tasks):
            n_deps = rng.randint(0, min(2, i))
            if n_deps > 0:
                possible = tasks[:i]
                deps[tasks[i]] = rng.sample(possible, min(n_deps, len(possible)))

        # Compute earliest start times
        earliest = {}
        for t in tasks:
            if not deps[t]:
                earliest[t] = 0
            else:
                earliest[t] = max(earliest[d] + durations[d] for d in deps[t])

        # Total time with all sequential
        total_seq = sum(durations.values())
        # Critical path (longest path)
        critical_time = max(earliest[t] + durations[t] for t in tasks)

        sidx = (self.seed or 0) % 16
        # Q&A
        if question_type == "total_sequential_time":
            question = _Q_SEQ_TIME[sidx]
            answer = total_seq
        elif question_type == "critical_path_time":
            question = _Q_CRIT_PATH[sidx]
            answer = critical_time
        elif question_type == "which_can_parallel":
            # Find tasks with no dependencies
            independent = [t for t in tasks if not deps[t]]
            if len(independent) < 2:
                question = f"Can any tasks start immediately (no dependencies)? If yes, which one?"
                answer = independent[0] if independent else tasks[0]
            else:
                question = _Q_PARALLEL[sidx]
                answer = len(independent)
        elif question_type == "earliest_start":
            # Pick a task with dependencies
            tasks_with_deps = [t for t in tasks if deps[t]]
            if not tasks_with_deps:
                return None
            target = rng.choice(tasks_with_deps)
            question = _Q_EARLIEST[sidx].format(target=target)
            answer = earliest[target]
        else:
            return None

        # Render
        style = self._random_style()
        palette = style["palette"]

        # Layout: topological sort, place left to right
        levels = {}
        for t in tasks:
            if not deps[t]:
                levels[t] = 0
            else:
                levels[t] = max(levels[d] for d in deps[t]) + 1

        max_level = max(levels.values())
        level_tasks = {l: [] for l in range(max_level + 1)}
        for t, l in levels.items():
            level_tasks[l].append(t)

        positions = {}
        for level, lt in level_tasks.items():
            x = level * 2.5 + 1
            for j, t in enumerate(lt):
                y = 2 + (j - len(lt) / 2) * 1.5
                positions[t] = (x, y)

        mode = pick_render_mode(sub_rng)
        if mode == "textbook":
            tbp = textbook_params(sub_rng)
            bg = tbp["bg"]
            arrow_col = tbp["aux_color"]
            node_edge = tbp["line_color"]
            text_color = "white"
            label_color = tbp["line_color"]
            font_kw = {"fontfamily": tbp["font_family"]}
            dpi = tbp["dpi"]
        elif mode == "sketch":
            bg = sub_rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"])
            arrow_col = "#555555"
            node_edge = "#1a1a1a"
            text_color = "white"
            label_color = "#2c3e50"
            font_kw = {}
            dpi = style["dpi"]
        else:
            bg = style["bg_color"]
            arrow_col = "#7f8c8d"
            node_edge = "black"
            text_color = "white"
            label_color = "#2c3e50"
            font_kw = {}
            dpi = style["dpi"]

        def _draw():
            fig, ax = plt.subplots(figsize=(max(8, n_tasks * 1.5), 5))
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)

            for t in tasks:
                for d in deps[t]:
                    dx, dy = positions[d]
                    tx, ty = positions[t]
                    ax.annotate("", xy=(tx - 0.4, ty), xytext=(dx + 0.4, dy),
                               arrowprops=dict(arrowstyle="->",
                                                color=arrow_col, lw=1.5))

            for t in tasks:
                x, y = positions[t]
                color = palette[levels[t] % len(palette)]
                circle = plt.Circle((x, y), 0.35, facecolor=color,
                                    edgecolor=node_edge,
                                    linewidth=1.5, alpha=0.8)
                ax.add_patch(circle)
                ax.text(x, y + 0.05, t, ha="center", va="center", fontsize=10,
                       fontweight="bold", color=text_color, **font_kw)
                ax.text(x, y - 0.55, f"d={durations[t]}", ha="center",
                        fontsize=8, color=label_color, **font_kw)

            ax.set_xlim(0, (max_level + 1) * 2.5 + 1)
            ax.set_ylim(-0.5, 5.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title("Task Dependency Graph (d = duration)",
                         fontsize=13, fontweight="bold", **font_kw)
            return fig

        if mode == "sketch":
            with sketch_context(scale=1.0, length=70, randomness=1.3):
                fig = _draw()
        else:
            fig = _draw()
        return question, str(answer), self.fig_to_pil(fig, dpi=dpi)
