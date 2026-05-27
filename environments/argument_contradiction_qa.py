"""
Argument Contradiction QA — given a short argument passage with multiple
statements, identify the statement that contradicts (or must be demonstrably
false given) the other statements in the passage.

The image renders an argument: a numbered list of premises + a multi-choice
question asking which option must be false. Output: A / B / C / D.

Difficulty axes:
  - number of premises (3 -> 7)
  - distractor strength (irrelevant -> "matches a different premise")
  - contradicted premise's distance from the contradicting fact (immediate vs
    chained)
"""
import random
import string
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Read the argument shown in the image. Taking each line in the argument to be true, which option must be demonstrably false? Answer with a single letter (A, B, C, or D) in <answer>...</answer>.",
    "Examine the argument in the image. Which statement (A, B, C, or D) contradicts the argument? Answer with one letter in <answer>...</answer>.",
    "The image shows a short argument with several premises. Which choice (A-D) is inconsistent with the argument? Output a single letter in <answer>...</answer>.",
    "Look at the passage in the image. Assume every line is true. Which of A, B, C, D must be false? Place a single letter in <answer>...</answer>.",
    "Read the argument shown. Which option contradicts the premises? Answer A, B, C, or D in <answer>...</answer>.",
    "The argument in the image lists several facts. Identify which option (A, B, C, D) cannot be true. Output a letter in <answer>...</answer>.",
    "Study the premises in the image. Which conclusion (A-D) is demonstrably false? Place one letter in <answer>...</answer>.",
    "Argument shown in image. Which option (A, B, C, D) contradicts what the argument says? Output a single letter in <answer>...</answer>.",
    "Read the numbered statements in the image. Which choice (A-D) directly contradicts at least one of them? Answer with a letter inside <answer>...</answer>.",
    "Examine the passage carefully. Which option (A, B, C, D) must be false if every premise in the argument is true? Output a letter in <answer>...</answer>.",
    "From the argument shown, identify the statement (A, B, C, or D) that is logically inconsistent with the premises. Place a single letter in <answer>...</answer>.",
    "The passage in the image presents several premises. Which choice (A-D) cannot follow from them? Output the letter in <answer>...</answer>.",
    "Argument analysis: which option (A, B, C, D) contradicts the passage in the image? Place a single letter in <answer>...</answer>.",
    "Look at the argument. Which statement (A-D) is demonstrably false given the premises? Output one letter in <answer>...</answer>.",
    "Read the image's argument. Pick the choice (A, B, C, or D) that conflicts with the stated facts. Place a letter in <answer>...</answer>.",
    "The image shows an argument plus four candidate statements. Which one contradicts the argument? Answer A/B/C/D in <answer>...</answer>.",
    "Premises in image. Which option (A-D) must be false? Output letter in <answer>...</answer>.",
]

# Scenario factories: each factory builds a list of premises and a contradicted
# statement plus distractors, parameterized by RNG and number of premises.

_AGE_NAMES = ["Anna", "Boris", "Cara", "Diego", "Elena", "Felix", "Gina",
              "Hugo", "Ivy", "Jorge", "Kira", "Liam"]
_PROF_NAMES = ["Sam", "Tina", "Uri", "Vera", "Walt", "Xena", "Yara", "Zane"]
_PROFESSIONS = ["doctor", "teacher", "engineer", "nurse", "chef", "pilot",
                 "lawyer", "writer"]
_CITY_NAMES = ["Aria", "Brent", "Cole", "Drew", "Erin", "Faye"]
_CITIES = ["Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Dakar", "Boston",
            "Sydney"]
_OBJECTS = ["coin", "key", "ring", "stamp", "pen", "card", "vase", "lamp"]
_BOX_LABELS = ["X", "Y", "Z", "W", "P", "Q", "R", "S"]
_COLORS = ["red", "blue", "green", "yellow", "purple", "black", "white",
            "orange"]


def _gen_age_scenario(rng, n_premises):
    """Generate an argument about ages where one option contradicts."""
    n_people = min(n_premises, 5)
    names = rng.sample(_AGE_NAMES, n_people)
    # Generate ages: distinct
    ages = rng.sample(range(15, 65), n_people)

    # Premises: pairwise comparisons, total comparisons can derive full order
    pairs = []
    # Always include n_people - 1 pairs that fully chain the order
    sorted_idx = sorted(range(n_people), key=lambda i: ages[i])  # ascending
    for k in range(n_people - 1):
        i = sorted_idx[k]
        j = sorted_idx[k + 1]
        # ages[i] < ages[j]: phrase "X is younger than Y" or "Y is older than X"
        if rng.random() < 0.5:
            pairs.append(f"{names[i]} is younger than {names[j]}.")
        else:
            pairs.append(f"{names[j]} is older than {names[i]}.")

    # Add absolute statements to pin some ages
    # Pin one age explicitly
    pin_idx = rng.randint(0, n_people - 1)
    pairs.append(f"{names[pin_idx]} is {ages[pin_idx]} years old.")

    # Possibly add a non-adjacent comparison for redundancy
    if n_premises >= 5 and n_people >= 3:
        i, j = rng.sample(range(n_people), 2)
        if ages[i] < ages[j]:
            pairs.append(f"{names[j]} is older than {names[i]}.")
        else:
            pairs.append(f"{names[i]} is older than {names[j]}.")

    # Trim/extend to exactly n_premises
    rng.shuffle(pairs)
    if len(pairs) > n_premises:
        pairs = pairs[:n_premises]
    while len(pairs) < n_premises:
        # Repeat with different phrasing
        i, j = rng.sample(range(n_people), 2)
        if ages[i] < ages[j]:
            pairs.append(f"{names[i]} is younger than {names[j]}.")
        else:
            pairs.append(f"{names[i]} is older than {names[j]}.")

    # Now build candidate statements
    # The contradicting statement should reverse a true ordering OR set wrong age
    p_correct = []  # statements consistent with argument (distractors)
    p_contradicting = []  # statements that contradict argument
    # True statements
    for k in range(n_people - 1):
        i = sorted_idx[k]; j = sorted_idx[k + 1]
        p_correct.append(f"{names[i]} is younger than {names[j]}.")
        p_correct.append(f"{names[j]} is older than {names[i]}.")
    # Statements about pinned name
    p_correct.append(f"{names[pin_idx]} is {ages[pin_idx]} years old.")
    # Contradicting statements
    for k in range(n_people - 1):
        i = sorted_idx[k]; j = sorted_idx[k + 1]
        p_contradicting.append(f"{names[i]} is older than {names[j]}.")
    # Wrong age for pinned
    wrong_age = ages[pin_idx] + rng.choice([-10, -8, -7, 7, 8, 10])
    p_contradicting.append(f"{names[pin_idx]} is {wrong_age} years old.")

    # Pick one contradiction
    contradiction = rng.choice(p_contradicting)
    # Pick three distractors (true / consistent statements not necessarily in
    # the premise text but consistent)
    rng.shuffle(p_correct)
    distractors = p_correct[:3]
    options = distractors + [contradiction]
    rng.shuffle(options)
    correct_letter = chr(ord("A") + options.index(contradiction))
    return pairs, options, correct_letter


def _gen_profession_scenario(rng, n_premises):
    """Build a profession-mapping argument: each person has a profession."""
    n_people = min(max(2, n_premises - 1), 4)
    names = rng.sample(_PROF_NAMES, n_people)
    profs = rng.sample(_PROFESSIONS, n_people)
    # mapping
    mapping = dict(zip(names, profs))

    premises = [f"{name} is a {prof}." for name, prof in mapping.items()]
    # Add a "not" clause
    other_prof = rng.choice([p for p in _PROFESSIONS if p not in profs])
    extra_name = rng.choice(names)
    premises.append(f"No {mapping[extra_name]} is a {other_prof}.")

    # Trim to n_premises
    rng.shuffle(premises)
    if len(premises) > n_premises:
        premises = premises[:n_premises]
    while len(premises) < n_premises:
        # add another consistent fact
        i, j = rng.sample(range(n_people), 2)
        premises.append(f"{names[i]} and {names[j]} have different professions.")

    # Candidate statements
    p_correct = [f"{name} is a {prof}." for name, prof in mapping.items()]
    # Negative correct: anyone not matched
    for name in names:
        for p in _PROFESSIONS:
            if p != mapping[name] and p in profs:
                p_correct.append(f"{name} is not a {p}.")
    # Contradiction: someone is a different profession than what the premise says
    contradictions = []
    for name in names:
        for p in _PROFESSIONS:
            if p != mapping[name]:
                contradictions.append(f"{name} is a {p}.")
    contradiction = rng.choice(contradictions)

    rng.shuffle(p_correct)
    distractors = p_correct[:3]
    options = distractors + [contradiction]
    rng.shuffle(options)
    correct_letter = chr(ord("A") + options.index(contradiction))
    return premises, options, correct_letter


def _gen_color_scenario(rng, n_premises):
    """Color-of-objects argument."""
    n_objs = min(max(2, n_premises - 1), 4)
    objs = rng.sample(_OBJECTS, n_objs)
    cols = rng.sample(_COLORS, n_objs)
    mapping = dict(zip(objs, cols))

    premises = [f"The {obj} is {col}." for obj, col in mapping.items()]
    # Add a count premise
    pin_obj = rng.choice(objs)
    pin_count = rng.randint(2, 5)
    premises.append(f"There are {pin_count} {mapping[pin_obj]} {pin_obj}s in total.")
    rng.shuffle(premises)
    if len(premises) > n_premises:
        premises = premises[:n_premises]
    while len(premises) < n_premises:
        # Add a non-color "comment" (fact about all/none)
        c = rng.choice(_COLORS)
        if c not in cols:
            premises.append(f"No {c} object is mentioned in the list.")
            break
        else:
            obj = rng.choice(objs)
            premises.append(f"The {obj} exists.")

    # Candidate statements
    p_correct = [f"The {obj} is {col}." for obj, col in mapping.items()]
    p_correct += [f"The {obj} is not {c}." for obj in objs for c in _COLORS
                  if c != mapping[obj] and c in cols]
    # Wrong count
    wrong_count = pin_count + rng.choice([-1, 1, 2])
    if wrong_count == pin_count:
        wrong_count += 1
    contradictions = [f"The {obj} is {c}." for obj in objs for c in _COLORS
                      if c != mapping[obj]]
    contradictions.append(
        f"There are {wrong_count} {mapping[pin_obj]} {pin_obj}s in total.")
    contradiction = rng.choice(contradictions)

    rng.shuffle(p_correct)
    distractors = p_correct[:3]
    options = distractors + [contradiction]
    rng.shuffle(options)
    correct_letter = chr(ord("A") + options.index(contradiction))
    return premises, options, correct_letter


def _gen_position_scenario(rng, n_premises):
    """Box positions on a shelf — e.g., 'X is to the left of Y'."""
    n = min(max(3, n_premises - 1), 5)
    labels = rng.sample(_BOX_LABELS, n)
    positions = list(range(n))
    rng.shuffle(positions)
    # positions[i] = position index of labels[i]
    pos_map = {labels[i]: positions[i] for i in range(n)}

    # Build sorted-order chain of premises
    sorted_labels = sorted(labels, key=lambda l: pos_map[l])
    premises = []
    for k in range(n - 1):
        a = sorted_labels[k]; b = sorted_labels[k + 1]
        premises.append(f"{a} is immediately to the left of {b}.")
    # Pin one position
    pin = rng.choice(labels)
    premises.append(f"{pin} is in position {pos_map[pin] + 1}.")

    rng.shuffle(premises)
    if len(premises) > n_premises:
        premises = premises[:n_premises]
    while len(premises) < n_premises:
        # Add general fact
        a, b = rng.sample(labels, 2)
        if pos_map[a] < pos_map[b]:
            premises.append(f"{a} is to the left of {b}.")
        else:
            premises.append(f"{a} is to the right of {b}.")

    # Candidate options
    p_correct = []
    for k in range(n - 1):
        a = sorted_labels[k]; b = sorted_labels[k + 1]
        p_correct.append(f"{a} is to the left of {b}.")
        p_correct.append(f"{b} is to the right of {a}.")
    p_correct.append(f"{pin} is in position {pos_map[pin] + 1}.")
    # Contradictions
    contradictions = []
    for k in range(n - 1):
        a = sorted_labels[k]; b = sorted_labels[k + 1]
        contradictions.append(f"{a} is to the right of {b}.")
        contradictions.append(f"{b} is to the left of {a}.")
    bad_pos = pos_map[pin] + rng.choice([-2, 2, -1, 1])
    if bad_pos == pos_map[pin]:
        bad_pos = pos_map[pin] + 1
    if 0 <= bad_pos < n:
        contradictions.append(f"{pin} is in position {bad_pos + 1}.")
    contradiction = rng.choice(contradictions)

    rng.shuffle(p_correct)
    distractors = p_correct[:3]
    options = distractors + [contradiction]
    rng.shuffle(options)
    correct_letter = chr(ord("A") + options.index(contradiction))
    return premises, options, correct_letter


_SCENARIO_GENERATORS = [
    _gen_age_scenario,
    _gen_profession_scenario,
    _gen_color_scenario,
    _gen_position_scenario,
]


class ArgumentContradictionQA(StandaloneVisualEnv):
    ENV_NAME = "argument_contradiction"

    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Be concise. Output just the letter (A, B, C, or D) inside `<answer>...</answer>`."
    )

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"n_premises": 3, "scenario_pool_size": 1}
        if level <= 2:
            return {"n_premises": 4, "scenario_pool_size": 2}
        if level <= 4:
            return {"n_premises": 5, "scenario_pool_size": 3}
        if level <= 6:
            return {"n_premises": 6, "scenario_pool_size": 4}
        if level <= 8:
            return {"n_premises": 7, "scenario_pool_size": 4}
        return {"n_premises": 8, "scenario_pool_size": 4}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4831 + level * 31 + 19)
        # Pool of scenario generators (ordered so L0 always uses first =
        # age, deterministic and easy)
        gens = _SCENARIO_GENERATORS[:cfg["scenario_pool_size"]]
        gen_fn = rng.choice(gens) if cfg["scenario_pool_size"] > 1 else gens[0]
        for _ in range(8):
            try:
                premises, options, correct_letter = gen_fn(rng,
                                                              cfg["n_premises"])
            except Exception:
                continue
            if len(options) != 4:
                continue
            template = rng.choice(_TEMPLATES)
            question = template
            img = self._render(premises, options, rng)
            return question, correct_letter, img
        return None

    def _render(self, premises, options, rng) -> Image.Image:
        bg = rng.choice(["#ffffff", "#fafafa", "#fffaf0", "#f5f9ff"])
        ff = rng.choice(["serif", "DejaVu Sans"])
        n_lines = len(premises) + len(options) + 4
        h = max(4.0, 0.42 * n_lines + 1.2)
        fig, ax = plt.subplots(figsize=(7.0, h), dpi=125)
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, n_lines + 0.5)
        ax.axis("off")

        # Border
        border = mpatches.FancyBboxPatch(
            (0.15, 0.2), 9.7, n_lines + 0.1,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=bg, edgecolor="#7f8c8d", linewidth=1.2)
        ax.add_patch(border)

        # Title
        title_y = n_lines + 0.05
        ax.text(5.0, title_y, "Argument", fontsize=14,
                 fontweight="bold", ha="center", va="top",
                 family=ff, color="#2c3e50")
        # Premises header
        y = title_y - 0.7
        ax.text(0.6, y, "Premises:", fontsize=12, fontweight="bold",
                 ha="left", va="top", family=ff, color="#34495e")
        y -= 0.42
        for i, p in enumerate(premises):
            ax.text(0.9, y, f"{i + 1}. {p}", fontsize=11,
                     ha="left", va="top", family=ff, color="#1a1a1a")
            y -= 0.42
        y -= 0.2
        ax.plot([0.7, 9.3], [y + 0.15, y + 0.15], color="#95a5a6",
                 linewidth=1.0)
        y -= 0.42
        ax.text(0.6, y, "Which is demonstrably false?",
                 fontsize=12, fontweight="bold", ha="left", va="top",
                 family=ff, color="#34495e")
        y -= 0.42
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            ax.text(0.9, y, f"({letter}) {opt}",
                     fontsize=11, ha="left", va="top", family=ff,
                     color="#1a1a1a")
            y -= 0.42

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                     facecolor=bg)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()


if __name__ == "__main__":
    env = ArgumentContradictionQA()
    for level in (0, 3, 6):
        for seed in (1, 7, 42):
            ok = env.generate(seed=seed, parameter={"level": level})
            print(f"L{level} s{seed} ok={ok} ans={env._answer}")
