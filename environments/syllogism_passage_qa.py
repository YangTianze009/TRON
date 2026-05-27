"""
Syllogism Passage QA environment (v3, redesigned 2026-04-16).

Goal: train classical deductive reasoning via syllogisms. The original env
was "text-heavy, barely visual" — the image was almost all text.

Redesign (v3):
  * v2 rendered premises as a plain text passage on white background.
  * v3 uses a MIX of visual representations for premises:
      - Venn diagrams (for "All/No/Some X are Y" premises)
      - Arrow flow charts (for "If A then B" hypothetical premises)
      - Icon pairs (a little cartoon of the premise subject and
        predicate linked by an arrow / barrier)
      - Circles-inside-circles diagrams for subset relations
      - Highlighted crossed-out diagrams for negations
  * Layouts:
      - Grid of mini-diagrams, one per premise.
      - Annotated flowchart (premises connected with arrows).
      - Mixed: some text premises + some visual diagrams.
  * The options box still contains text but uses different font styles,
    rotated slightly, randomized backgrounds.
  * 6 question templates.
  * Title variants; colour palette randomized per seed.
  * L0 vs L9 structural shift:
      - L0: 3 premises all as Venn diagrams, clean fonts.
      - L9: 6-8 premises mixing diagrams + text + red-herrings, cramped
        but readable layout.
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

# Neutral category names
_NEUTRAL_CATEGORIES = [
    "meters", "inches", "yards",
    "glimmers", "drakes", "valerions",
    "quints", "petroxes", "crinoids",
    "salfas", "hendrons", "terrapins",
    "ulams", "qorans", "whelkins",
    "phenomena", "phenomenae", "phenomeni",
    "trellons", "bryndals", "klixons",
    "aspirants", "ozarks", "merkins",
    "hypersols", "granates", "embers",
    "quasites", "rexors", "lemvigs",
    "ternites", "berzons", "fossets",
]

# Colour palettes for the little Venn diagrams
_VENN_PALETTES = [
    ("#3498db", "#e74c3c"),
    ("#27ae60", "#8e44ad"),
    ("#f39c12", "#16a085"),
    ("#c0392b", "#2980b9"),
    ("#2c3e50", "#d35400"),
    ("#5dade2", "#ec7063"),
    ("#1abc9c", "#f1c40f"),
    ("#34495e", "#e67e22"),
]

_BG_COLORS = [
    "#ffffff", "#fefae0", "#e8f4f8", "#fdf2e9",
    "#f4ecf7", "#e8f8f5", "#fef5e7",
]

# ------------------------------------------------------------------ #
# Helpers for syllogism scaffolds (kept from v2)
# ------------------------------------------------------------------ #

def _pick_categories(rng: random.Random, k: int) -> List[str]:
    return rng.sample(_NEUTRAL_CATEGORIES, k)

def _transitive_subset_chain(rng: random.Random, cats: List[str]) -> Dict:
    premises = []
    for i in range(len(cats) - 1):
        premises.append({"text": f"All {cats[i]} are {cats[i + 1]}.",
                          "kind": "all",
                          "subj": cats[i],
                          "pred": cats[i + 1]})
    A, Z = cats[0], cats[-1]
    mid = cats[len(cats) // 2]
    valid = f"All {A} are {Z}."
    fallacies = {
        "converse_of_valid":    f"All {Z} are {A}.",
        "negation":             f"No {A} are {Z}.",
        "negation_reverse":     f"No {Z} are {A}.",
        "mid_chain_converse":   f"All {mid} are {A}.",
        "all_mid_reverse":      f"All {Z} are {mid}.",
        "some_not_mid":         f"Some {A} are not {mid}.",
    }
    return {
        "premises": premises,
        "valid_conclusion": valid,
        "fallacies": fallacies,
        "structure": "barbara_chain",
    }

def _mixed_some_chain(rng: random.Random, cats: List[str]) -> Dict:
    X, A, B, C, D = cats
    premises = [
        {"text": f"All {A} are {B}.", "kind": "all", "subj": A, "pred": B},
        {"text": f"All {B} are {C}.", "kind": "all", "subj": B, "pred": C},
        {"text": f"All {C} are {D}.", "kind": "all", "subj": C, "pred": D},
        {"text": f"Some {X} are {A}.", "kind": "some", "subj": X,
         "pred": A},
    ]
    valid = f"Some {X} are {D}."
    fallacies = {
        "strengthening":       f"All {X} are {D}.",
        "illicit_particular":  f"Some {D} are not {X}.",
        "contradicted":        f"No {X} are {D}.",
        "some_x_not_d":        f"Some {X} are not {D}.",
        "all_d_are_x":         f"All {D} are {X}.",
        "no_d_are_x":          f"No {D} are {X}.",
    }
    return {
        "premises": premises,
        "valid_conclusion": valid,
        "fallacies": fallacies,
        "structure": "mixed_some_chain",
    }

def _chain_with_neg(rng: random.Random, cats: List[str]) -> Dict:
    X, A, B, C = cats
    premises = [
        {"text": f"All {X} are {A}.", "kind": "all", "subj": X, "pred": A},
        {"text": f"All {A} are {B}.", "kind": "all", "subj": A, "pred": B},
        {"text": f"No {B} are {C}.", "kind": "no", "subj": B, "pred": C},
    ]
    valid = f"No {X} are {C}."
    fallacies = {
        "flipped_contradict":  f"All {X} are {C}.",
        "illicit_particular":  f"Some {X} are {C}.",
        "mid_converse":        f"All {A} are {X}.",
        "scope_error":         f"All {C} are {X}.",
        "bad_some":            f"Some {C} are {X}.",
        "wrong_direction":     f"All {B} are {X}.",
    }
    return {
        "premises": premises,
        "valid_conclusion": valid,
        "fallacies": fallacies,
        "structure": "chain_with_neg",
    }

def _hypothetical_chain(rng: random.Random, n_steps: int) -> Dict:
    steps_templates = [
        (f"the alarm rings", f"the dog barks", f"the cat hides",
         f"the bird flies", f"the tree shakes", f"the leaves fall"),
        (f"it rains", f"the roof gets wet", f"the gutters fill",
         f"the drain overflows", f"the garden floods", f"the path erodes"),
        (f"the lamp is on", f"the desk is lit", f"the papers are visible",
         f"the writing is clear", f"the edit is done", f"the draft is sent"),
    ]
    chain = list(random.Random(n_steps + 7).choice(steps_templates))
    chain = chain[:n_steps + 1]
    premises = []
    for i in range(n_steps):
        premises.append({
            "text": f"If {chain[i]}, then {chain[i + 1]}.",
            "kind": "if",
            "ante": chain[i],
            "cons": chain[i + 1],
        })
    A, Z = chain[0], chain[-1]
    valid = f"If {A}, then {Z}."
    mid = chain[n_steps // 2] if n_steps >= 2 else chain[0]
    fallacies = {
        "affirm_consequent":   f"If {Z}, then {A}.",
        "deny_antecedent":     f"If not {A}, then not {Z}.",
        "mid_chain_converse":  f"If {mid}, then {A}.",
        "contradicted":        f"If {A}, then not {Z}.",
        "unrelated":           f"{A.capitalize()} and {Z} are unrelated.",
        "reverse_chain":       f"If {Z}, then {mid}.",
    }
    return {
        "premises": premises,
        "valid_conclusion": valid,
        "fallacies": fallacies,
        "structure": "hypothetical_chain",
    }

def _red_herring_premise(rng: random.Random, used_cats: set) -> Optional[Dict]:
    unused = [c for c in _NEUTRAL_CATEGORIES if c not in used_cats]
    if len(unused) < 2:
        return None
    a, b = rng.sample(unused, 2)
    template_kind = rng.choice(["all", "some", "no", "some_not"])
    if template_kind == "all":
        return {"text": f"All {a} are {b}.", "kind": "all",
                "subj": a, "pred": b}
    if template_kind == "some":
        return {"text": f"Some {a} are {b}.", "kind": "some",
                "subj": a, "pred": b}
    if template_kind == "no":
        return {"text": f"No {a} are {b}.", "kind": "no",
                "subj": a, "pred": b}
    return {"text": f"Some {a} are not {b}.", "kind": "some_not",
            "subj": a, "pred": b}

# ------------------------------------------------------------------ #
# Visual rendering of premises
# ------------------------------------------------------------------ #

def _render_venn_all(ax, subj: str, pred: str, palette):
    """Subject circle inscribed inside predicate circle."""
    c1, c2 = palette
    # Outer circle (pred)
    outer = mpatches.Circle((0.5, 0.5), 0.34, fc=c2, ec="#2c3e50",
                            lw=1.5, alpha=0.28)
    ax.add_patch(outer)
    # Inner (subj)
    inner = mpatches.Circle((0.45, 0.48), 0.18, fc=c1, ec="#2c3e50",
                            lw=1.5, alpha=0.55)
    ax.add_patch(inner)
    ax.text(0.45, 0.48, subj, fontsize=9, ha="center", va="center",
            color="#1a1a1a", fontweight="bold")
    ax.text(0.5, 0.88, pred, fontsize=9, ha="center", va="center",
            color="#1a1a1a", fontweight="bold")

def _render_venn_some(ax, subj: str, pred: str, palette):
    """Overlapping circles with shared part highlighted."""
    c1, c2 = palette
    c_a = mpatches.Circle((0.38, 0.5), 0.22, fc=c1, ec="#2c3e50",
                          lw=1.5, alpha=0.35)
    c_b = mpatches.Circle((0.62, 0.5), 0.22, fc=c2, ec="#2c3e50",
                          lw=1.5, alpha=0.35)
    ax.add_patch(c_a)
    ax.add_patch(c_b)
    # Dot in intersection
    ax.add_patch(mpatches.Circle((0.5, 0.5), 0.035, fc="#2c3e50",
                                 ec="#2c3e50", lw=0.8))
    ax.text(0.5, 0.38, "\u2717", fontsize=12, ha="center", va="center",
            color="#2c3e50")
    # Separate labels vertically (above + below) to prevent overlap.
    ax.text(0.5, 0.9, _truncate(subj, 14), fontsize=8, ha="center",
            va="center", color="#1a1a1a", fontweight="bold")
    ax.text(0.5, 0.12, _truncate(pred, 14), fontsize=8, ha="center",
            va="center", color="#1a1a1a", fontweight="bold")

def _render_venn_no(ax, subj: str, pred: str, palette):
    """Disjoint circles."""
    c1, c2 = palette
    c_a = mpatches.Circle((0.3, 0.5), 0.18, fc=c1, ec="#2c3e50",
                          lw=1.5, alpha=0.35)
    c_b = mpatches.Circle((0.7, 0.5), 0.18, fc=c2, ec="#2c3e50",
                          lw=1.5, alpha=0.35)
    ax.add_patch(c_a)
    ax.add_patch(c_b)
    # Big NO
    ax.plot([0.5, 0.5], [0.32, 0.68], color="#e74c3c", lw=2.5)
    ax.plot([0.42, 0.58], [0.42, 0.58], color="#e74c3c", lw=2.5)
    # Place labels on two separate lines to prevent horizontal overlap.
    ax.text(0.22, 0.85, _truncate(subj, 12), fontsize=8, ha="left",
            va="center", color="#1a1a1a", fontweight="bold")
    ax.text(0.22, 0.14, _truncate(pred, 12), fontsize=8, ha="left",
            va="center", color="#1a1a1a", fontweight="bold")

def _render_venn_some_not(ax, subj: str, pred: str, palette):
    """Subj circle partly outside pred circle."""
    c1, c2 = palette
    c_pred = mpatches.Circle((0.55, 0.5), 0.26, fc=c2, ec="#2c3e50",
                             lw=1.5, alpha=0.3)
    c_subj = mpatches.Circle((0.4, 0.48), 0.22, fc=c1, ec="#2c3e50",
                             lw=1.5, alpha=0.35)
    ax.add_patch(c_pred)
    ax.add_patch(c_subj)
    # X in non-overlap region
    ax.text(0.25, 0.42, "\u2717", fontsize=12, ha="center", va="center",
            color="#e74c3c", fontweight="bold")
    ax.text(0.5, 0.9, f"{_truncate(subj, 10)} (subj)", fontsize=8,
            ha="center", va="center", color="#1a1a1a", fontweight="bold")
    ax.text(0.5, 0.12, f"{_truncate(pred, 10)} (pred)", fontsize=8,
            ha="center", va="center", color="#1a1a1a", fontweight="bold")

def _render_if_flow(ax, ante: str, cons: str, palette):
    """A -> B arrow with mini boxes."""
    c1, c2 = palette
    # Box A
    rect1 = mpatches.FancyBboxPatch(
        (0.02, 0.35), 0.42, 0.3,
        boxstyle="round,pad=0.03",
        facecolor=c1, edgecolor="#2c3e50", lw=1.2, alpha=0.4)
    ax.add_patch(rect1)
    ax.text(0.23, 0.5, _truncate(ante, 18), fontsize=8, ha="center",
            va="center", wrap=True, color="#1a1a1a", fontweight="bold")
    # Arrow
    ax.annotate("", xy=(0.56, 0.5), xytext=(0.46, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50"))
    # Box B
    rect2 = mpatches.FancyBboxPatch(
        (0.58, 0.35), 0.4, 0.3,
        boxstyle="round,pad=0.03",
        facecolor=c2, edgecolor="#2c3e50", lw=1.2, alpha=0.4)
    ax.add_patch(rect2)
    ax.text(0.78, 0.5, _truncate(cons, 18), fontsize=8, ha="center",
            va="center", wrap=True, color="#1a1a1a", fontweight="bold")

def _render_text_premise(ax, text: str):
    ax.text(0.5, 0.5, text, fontsize=10, ha="center", va="center",
            color="#1a1a1a", wrap=True)

def _truncate(s, n):
    if len(s) > n:
        return s[:n - 1] + "\u2026"
    return s

def _render_premise_visual(ax, premise: Dict, palette):
    kind = premise.get("kind", "text")
    if kind == "all":
        _render_venn_all(ax, premise["subj"], premise["pred"], palette)
    elif kind == "some":
        _render_venn_some(ax, premise["subj"], premise["pred"], palette)
    elif kind == "no":
        _render_venn_no(ax, premise["subj"], premise["pred"], palette)
    elif kind == "some_not":
        _render_venn_some_not(ax, premise["subj"], premise["pred"],
                              palette)
    elif kind == "if":
        _render_if_flow(ax, premise["ante"], premise["cons"], palette)
    else:
        _render_text_premise(ax, premise.get("text", ""))

# ------------------------------------------------------------------ #
# Env class
# ------------------------------------------------------------------ #

class SyllogismPassageQA(StandaloneVisualEnv):
    ENV_NAME = "syllogism_passage"

    _QUESTION_TEMPLATES = [
        "Read the premises shown as Venn diagrams and text in the image. Based ONLY on the premises shown, which conclusion follows logically? Answer with a single letter (A, B, C, or D).",
        "The image presents a set of logical premises as visual diagrams (Venn circles for categorical statements, arrow flows for conditionals). Which of the four conclusions can be validly deduced? Answer with a single letter.",
        "Study the premise diagrams displayed in the image. Which conclusion necessarily follows from them? Answer with a single letter A-D.",
        "Given only the premises shown in the image, which option is a logically valid conclusion? Answer with a single letter.",
        "Inspect the visually-rendered premises (Venn diagrams and conditional flows) and determine the valid conclusion among the options. Answer with a single letter.",
        "The premises in the image use Venn-like circles and arrow diagrams. Deduce the valid conclusion and answer with a single letter.",
    ]

    _TITLE_VARIANTS = [
        "Logic Problem",
        "Deductive Reasoning",
        "Logical Argument",
        "Syllogism",
        "Premises & Conclusion",
        "Deduction",
        "Visual Logic",
    ]

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_premises = 3 + level // 2
        n_red_herrings = 1 + (level + 1) // 2
        n_near_valid_distractors = min(3, 2 + level // 6)
        # Visual rendering style
        # L0-L1: all premises as diagrams
        # L3+: mix diagrams + text
        # L6+: allow cramped grid
        if level <= 1:
            vis_mode = "all_visual"
        elif level <= 5:
            vis_mode = "mixed"
        else:
            vis_mode = "mixed_dense"
        return {
            "n_premises": n_premises,
            "n_red_herrings": n_red_herrings,
            "n_near_valid_distractors": n_near_valid_distractors,
            "use_contradictory_subset": level >= 8,
            "vis_mode": vis_mode,
        }

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[
            Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 31 + level * 7 + 11)
        self._primary_complexity_feature = cfg["n_premises"]
        for _ in range(30):
            result = self._try_generate(rng, level, cfg)
            if result is not None:
                return result
        return None

    def _try_generate(self, rng, level, cfg):
        n_premises = cfg["n_premises"]
        if level <= 3:
            scaffolds = ["barbara", "mixed_some", "hypothetical"]
        else:
            scaffolds = ["barbara", "mixed_some", "chain_neg",
                         "hypothetical"]
        scaffold = rng.choice(scaffolds)

        if scaffold == "barbara":
            n_cats = min(6, max(3, n_premises - cfg["n_red_herrings"] + 1))
            if n_cats < 3:
                n_cats = 3
            cats = _pick_categories(rng, n_cats)
            syl = _transitive_subset_chain(rng, cats)
            used_cats = set(cats)
        elif scaffold == "mixed_some":
            cats = _pick_categories(rng, 5)
            syl = _mixed_some_chain(rng, cats)
            used_cats = set(cats)
        elif scaffold == "chain_neg":
            cats = _pick_categories(rng, 4)
            syl = _chain_with_neg(rng, cats)
            used_cats = set(cats)
        else:
            n_steps = max(2, min(5, n_premises - cfg["n_red_herrings"]))
            syl = _hypothetical_chain(rng, n_steps)
            used_cats = set()

        red_herrings = []
        for _ in range(cfg["n_red_herrings"]):
            rh = _red_herring_premise(rng, used_cats)
            if rh:
                red_herrings.append(rh)

        all_premises = list(syl["premises"]) + red_herrings
        rng.shuffle(all_premises)

        correct = syl["valid_conclusion"]
        fallacies = list(syl["fallacies"].values())
        rng.shuffle(fallacies)
        fallacies = [f for f in fallacies if f != correct]
        if len(fallacies) < 3:
            return None
        n_near = cfg["n_near_valid_distractors"]
        near_valid = fallacies[:n_near]
        while len(near_valid) < 3:
            near_valid.append(fallacies[len(near_valid) % len(fallacies)])
        distractors = near_valid[:3]
        options = list(distractors)
        insert_idx = rng.randint(0, 3)
        options.insert(insert_idx, correct)
        answer_letter = chr(ord("A") + insert_idx)

        title = rng.choice(self._TITLE_VARIANTS)
        image = self._render_passage(all_premises, options, title=title,
                                     rng=rng, cfg=cfg)
        question = rng.choice(self._QUESTION_TEMPLATES)
        return question, answer_letter, image

    # -------------------------------------------------- #
    # Rendering
    # -------------------------------------------------- #

    def _render_passage(self, premises: List[Dict], options: List[str],
                        title: str, rng: random.Random,
                        cfg: Dict) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        bg = rng.choice(_BG_COLORS)
        font_fam = rng.choice(["serif", "DejaVu Sans", "sans-serif"])

        n_prem = len(premises)
        n_cols = 3 if n_prem <= 6 else 4
        n_rows = (n_prem + n_cols - 1) // n_cols

        fig_w = 10.0 * sc
        fig_h = (2.0 + 1.5 * n_rows + 1.5 + 0.5 * len(options)) * sc
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(bg)

        # Title
        fig.text(0.5, 0.98, title, fontsize=16, fontweight="bold",
                 ha="center", va="top", family=font_fam,
                 color="#2c3e50")
        fig.text(0.5, 0.945, "Premises:", fontsize=12,
                 fontweight="bold", ha="center", color="#34495e",
                 family=font_fam)

        # Premise diagrams
        top_area = 0.93
        bottom_area = 0.40 + 0.03 * len(options)
        grid_height = top_area - bottom_area
        for i, prem in enumerate(premises):
            r = i // n_cols
            c = i % n_cols
            left = 0.03 + c * (0.94 / n_cols)
            w = 0.90 / n_cols
            h = grid_height / max(n_rows, 1) * 0.92
            bottom = top_area - (r + 1) * (grid_height / max(n_rows, 1))
            ax = fig.add_axes([left, bottom, w, h])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.axis("off")
            # Frame
            rect = mpatches.Rectangle((0.02, 0.03), 0.96, 0.94,
                                       facecolor="#ffffff",
                                       edgecolor="#95a5a6",
                                       linewidth=1.0, zorder=0)
            ax.add_patch(rect)
            # Alternate visual vs text style by mode
            mode = cfg["vis_mode"]
            if mode == "all_visual":
                do_visual = True
            elif mode == "mixed":
                do_visual = (i % 2 == 0) or prem.get("kind") in ("if",)
                # Ensure most complex premises get visual treatment
            else:
                do_visual = (rng.random() < 0.7)
            if do_visual:
                palette = rng.choice(_VENN_PALETTES)
                _render_premise_visual(ax, prem, palette)
            else:
                ax.text(0.5, 0.5, prem.get("text", ""), fontsize=10,
                        ha="center", va="center", color="#1a1a1a",
                        family=font_fam, wrap=True)
            ax.text(0.03, 0.97, f"{i + 1}", fontsize=10,
                    fontweight="bold", ha="left", va="top",
                    color="#2c3e50")

        # Options section
        opt_top = bottom_area - 0.02
        opt_bottom = 0.02
        ax_opts = fig.add_axes([0.05, opt_bottom, 0.90,
                                 opt_top - opt_bottom])
        ax_opts.set_xlim(0, 10)
        ax_opts.set_ylim(0, 1)
        ax_opts.axis("off")
        ax_opts.text(0.1, 0.92, "Which conclusion follows?",
                     fontsize=13, fontweight="bold", ha="left",
                     va="top", color="#34495e", family=font_fam)
        # Divider line
        ax_opts.plot([0.1, 9.9], [0.84, 0.84], color="#95a5a6",
                     linewidth=1.0)
        opt_step = 0.80 / len(options)
        for i, opt in enumerate(options):
            letter = chr(ord("A") + i)
            y = 0.84 - (i + 1) * opt_step
            ax_opts.text(0.2, y, f"({letter}) {opt}",
                         fontsize=12, ha="left", va="center",
                         family=font_fam, color="#1a1a1a")

        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    import collections
    out_dir = "/tmp/env_check"
    os.makedirs(out_dir, exist_ok=True)
    env = SyllogismPassageQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            ok = env.generate(seed=seed, parameter={"level": level})
            if not ok:
                print(f"[seed={seed} L{level}] FAILED to generate")
                continue
            print(f"[seed={seed} L{level}] A={env._answer}")
    for level in (0, 3, 6, 9):
        letters = collections.Counter()
        for s in range(20):
            e = SyllogismPassageQA()
            ok = e.generate(seed=s * 1000 + level * 37 + 17,
                            parameter={"level": level})
            if ok:
                letters[e._answer] += 1
        print(f"[L{level}] letters={dict(letters)}")
