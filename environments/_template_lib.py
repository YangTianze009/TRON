"""Shared template library for Part E1 (template diversity expansion).

Envs with < 16 unique templates can import from here to reach the floor
without inventing phrasings one-off. Each function returns ≥ 16 distinct
templates built along the axes in
`environments/detected_bugs/2026-04-22-E-template-recipe.md`.

Usage:
    from ._template_lib import count_templates, mcq_templates
    template_pool = count_templates(noun="filled small triangle")
    question = rng.choice(template_pool)

The templates must reference the figure / image (not inline numerics) so
the model is forced to look at the rendered image. See anti-patterns
section of the recipe doc.
"""
from typing import List

# ------------------------------------------------------------------ #
# Count-type questions (integer answer)
# ------------------------------------------------------------------ #

def count_templates(noun: str, plural_noun: str = None) -> List[str]:
    """16 phrasings for 'how many <noun>s are in the figure'.

    Args:
        noun: singular noun, e.g. "triangle", "filled small square",
              "colored cell".
        plural_noun: optional; defaults to noun+"s".
    """
    n = noun
    ns = plural_noun if plural_noun else noun + "s"
    return [
        # Axis A: direct interrogative (4)
        f"How many {ns} are shown in the figure?",
        f"In the image, how many {ns} are visible?",
        f"Looking at the figure, how many {ns} can you count?",
        f"The figure contains how many {ns}?",
        # Axis A: imperative (4)
        f"Count the {ns} in the figure. Answer with a single integer.",
        f"Enumerate the {ns} visible in the image. State the total.",
        f"Tally the {ns} in the figure. Give one integer.",
        f"Report the number of {ns} shown.",
        # Axis A: declarative / cloze (4)
        f"The number of {ns} in the figure is ___. Fill in the blank with a single integer.",
        f"Let N denote the count of {ns} in the figure. Compute N.",
        f"The figure shows N {ns}. Find N.",
        f"In this image there are N {ns}. What is N?",
        # Axis A: reasoning-prompt (4)
        f"Examine the figure carefully and determine the number of {ns} present.",
        f"Identify every {n} in the image and state the total count.",
        f"Reason about the figure to find the number of {ns} shown.",
        f"After inspecting the image, report how many {ns} appear.",
    ]

# ------------------------------------------------------------------ #
# MCQ with letter answer (A, B, C, D...)
# ------------------------------------------------------------------ #

def mcq_templates(task_clause: str, n_choices: int = 4) -> List[str]:
    """16 phrasings for MCQ with letter answer.

    Args:
        task_clause: active-voice clause describing the task, without
                     options, e.g. "identifies the missing piece" or
                     "matches the rotated shape".
        n_choices: number of choices (default 4 → A-D).
    """
    last_letter = chr(ord("A") + n_choices - 1)
    letters = f"A-{last_letter}"
    letters_comma = ", ".join(chr(ord("A") + i) for i in range(n_choices))
    return [
        # Interrogative + answer-format hint (4)
        f"Which option {task_clause}? Answer with a single letter ({letters}).",
        f"Which of the choices ({letters}) {task_clause}? Reply with one letter only.",
        f"Which candidate {task_clause}? Give the letter: {letters_comma}.",
        f"Which of the following {task_clause}? Answer: {letters_comma}.",
        # Imperative (4)
        f"Select the option that {task_clause}. Letter only ({letters}).",
        f"Pick the choice that {task_clause}. Reply {letters}.",
        f"Choose the correct option from {letters_comma}.",
        f"Identify the choice that {task_clause} and answer with its letter.",
        # Declarative / cloze (4)
        f"The correct option is ___. Fill in with a letter from {letters_comma}.",
        f"The option that {task_clause} is labelled ___. Give one letter.",
        f"Let X be the letter of the correct option. Then X = ?",
        f"The figure's correct match is option ___. Provide {letters}.",
        # Reasoning-prompt (4)
        f"Examine each candidate and determine which one {task_clause}. Answer with a single letter ({letters}).",
        f"Consider the options. Which one {task_clause}? Reply with {letters_comma}.",
        f"Reason about the figure and each choice, then give the letter ({letters}) of the correct option.",
        f"After comparing all candidates, state which one {task_clause} (answer: a single letter {letters_comma}).",
    ]

# ------------------------------------------------------------------ #
# Measurement / numeric value ("what is X")
# ------------------------------------------------------------------ #

def measure_templates(quantity_phrase: str, unit: str = "",
                      rounding: str = "") -> List[str]:
    """16 phrasings for 'what is the <quantity>' with numeric answer.

    Args:
        quantity_phrase: e.g. "area of the shaded region", "length AB",
                         "measure of angle BAC".
        unit: optional, e.g. "degrees", "units".
        rounding: optional trailing hint, e.g. "Round to 2 decimals."
    """
    q = quantity_phrase
    u = f" in {unit}" if unit else ""
    r = f" {rounding}" if rounding else ""
    return [
        # Direct interrogative (4)
        f"What is the {q}{u}?{r}",
        f"In the figure, what is the {q}{u}?{r}",
        f"Based on the image, what is the value of the {q}{u}?{r}",
        f"What is the {q}{u} in the figure shown?{r}",
        # Imperative (4)
        f"Find the {q}{u} in the figure.{r}",
        f"Compute the {q}{u}.{r}",
        f"Determine the {q}{u} based on the image.{r}",
        f"Calculate the {q}{u}.{r}",
        # Declarative / cloze (4)
        f"The {q}{u} is ___.{r}",
        f"The figure shows that the {q}{u} equals ___.{r}",
        f"In the image, the {q}{u} is equal to ___.{r}",
        f"Let X be the {q}{u}. Compute X.{r}",
        # Reasoning-prompt (4)
        f"Examine the figure and find the {q}{u}.{r}",
        f"Using the information in the image, determine the {q}{u}.{r}",
        f"Reason about the figure to compute the {q}{u}.{r}",
        f"Inspect the figure and state the {q}{u}.{r}",
    ]

# ------------------------------------------------------------------ #
# Comparison: which of X or Y is <property>
# ------------------------------------------------------------------ #

def comparison_templates(property_adj: str = "larger",
                         letter_a: str = "A",
                         letter_b: str = "B") -> List[str]:
    """16 phrasings for binary comparison ('which is bigger').

    Args:
        property_adj: e.g. "larger", "taller", "heavier", "shorter".
        letter_a, letter_b: item labels.
    """
    p = property_adj
    a, b = letter_a, letter_b
    return [
        # Direct interrogative (4)
        f"Which of {a} and {b} is {p}?",
        f"Between {a} and {b}, which is {p}?",
        f"In the figure, which one ({a} or {b}) is {p}?",
        f"Which is {p}, {a} or {b}?",
        # Imperative (4)
        f"Identify the {p} one between {a} and {b}.",
        f"Determine which is {p}: {a} or {b}.",
        f"Choose the {p} of {a} and {b}.",
        f"Report whether {a} or {b} is {p}.",
        # Declarative / cloze (4)
        f"The {p} of {a} and {b} is ___.",
        f"Between {a} and {b}, the {p} item is ___.",
        f"The item with the {p} property is ___.",
        f"Let X be the {p} of ({a}, {b}). Then X = ?",
        # Reasoning-prompt (4)
        f"Examine {a} and {b} in the figure and state which is {p}.",
        f"Compare {a} and {b} and indicate which is {p}.",
        f"Reason about the figure to decide whether {a} or {b} is {p}.",
        f"Inspect both options and report the {p} one.",
    ]

# ------------------------------------------------------------------ #
# Yes / No
# ------------------------------------------------------------------ #

def yesno_templates(claim_clause: str) -> List[str]:
    """16 phrasings for yes/no question about a claim.

    Args:
        claim_clause: e.g. "the two triangles are congruent",
                      "the shape has rotational symmetry of order 3".
    """
    c = claim_clause
    return [
        # Direct interrogative (4)
        f"Is it true that {c}?",
        f"Does the figure show that {c}?",
        f"In the image, is {c} the case?",
        f"Given the figure, does {c}?",
        # Imperative (4)
        f"Decide whether {c}. Answer yes or no.",
        f"Determine if {c}. Reply yes or no.",
        f"State whether {c}. (yes / no)",
        f"Confirm or deny: {c}. Answer yes or no.",
        # Declarative / cloze (4)
        f"The claim '{c}' is ___. (yes / no)",
        f"It is ___ that {c}. (yes / no)",
        f"Whether {c} is ___ (yes / no).",
        f"Let P be the truth of the claim '{c}'. Report P as yes or no.",
        # Reasoning-prompt (4)
        f"Examine the figure and determine whether {c}. Answer yes or no.",
        f"Reason about the figure to decide if {c}. Yes or no?",
        f"Inspect the image. Does {c}? Reply yes or no.",
        f"Based on the figure, is {c}? yes/no.",
    ]

# ------------------------------------------------------------------ #
# Match / identify (pick the one matching some criterion)
# ------------------------------------------------------------------ #

def match_templates(target_desc: str, n_options: int = 4) -> List[str]:
    """16 phrasings for 'which option matches the target'.

    Args:
        target_desc: e.g. "the shape after rotation", "the original
                     pattern", "the 3D solid shown in the image".
        n_options: number of options.
    """
    last = chr(ord("A") + n_options - 1)
    letters = f"A-{last}"
    t = target_desc
    return [
        # Direct interrogative (4)
        f"Which option matches {t}? Answer {letters}.",
        f"Which of the choices is equivalent to {t}?",
        f"Which candidate corresponds to {t}? ({letters})",
        f"Which option best represents {t}?",
        # Imperative (4)
        f"Select the option that matches {t}. Reply with a single letter.",
        f"Pick the choice representing {t}. Letter only.",
        f"Identify the candidate that matches {t}. ({letters})",
        f"Choose the option equivalent to {t}.",
        # Declarative / cloze (4)
        f"The option matching {t} is ___. ({letters})",
        f"{t.capitalize()} corresponds to option ___. ({letters})",
        f"Let X be the letter of the option matching {t}. X = ?",
        f"The candidate that represents {t} is ___. Give one letter.",
        # Reasoning-prompt (4)
        f"Examine {t} and each candidate, then select the matching one ({letters}).",
        f"Compare each option to {t} and identify the match.",
        f"Reason about which candidate corresponds to {t}.",
        f"Inspect {t} and find its match among the options ({letters}).",
    ]

# ------------------------------------------------------------------ #
# Title pools (for ax.set_title)
# ------------------------------------------------------------------ #

TITLE_POOLS = {
    "geometry": [
        "Figure", "Diagram", "Given Figure", "Construction", "Problem Figure",
        "Geometric Configuration", "Problem Diagram", "Geometry Figure",
    ],
    "fractal": [
        "Fractal Pattern", "Iterated Pattern", "Self-similar Structure",
        "Recursive Pattern", "Fractal Figure", "Scaling Pattern",
        "Fractal Construction", "Iterated Fractal",
    ],
    "grid_puzzle": [
        "Pattern Grid", "Color Grid", "Grid Puzzle", "Tile Puzzle",
        "Missing Piece Problem", "Jigsaw Grid", "Grid Completion",
        "Puzzle Grid",
    ],
    "chart": [
        "Data Chart", "Chart", "Figure", "Data Visualization", "Plot",
        "Bar Chart", "Graph", "Data Plot",
    ],
    "3d": [
        "3D Figure", "Solid Shape", "3D Construction", "Polycube",
        "3D Object", "Volumetric Figure", "Solid", "Block Structure",
    ],
    "graph": [
        "Graph", "Network", "Node Diagram", "Graph Diagram",
        "Vertex-Edge Graph", "Connection Graph", "Network Diagram",
        "Graph Figure",
    ],
    "rotation": [
        "Rotation Problem", "Rotated Shape", "Rotation Figure",
        "Rotational Configuration", "Spin Problem", "Rotation Task",
        "Oriented Figure", "Rotation Diagram",
    ],
}

def title_pool(category: str) -> List[str]:
    """Return the title pool for a category, falling back to 'geometry'."""
    return TITLE_POOLS.get(category, TITLE_POOLS["geometry"])
