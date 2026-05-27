"""Shared rendering mode helpers for Part E2 (rendering realism).

Provides three rendering "textures" that any env can opt into:
  - clean:     current behavior (bright palettes, sans-serif, saturated)
  - textbook:  math textbook feel (serif font, thin grey lines, white bg)
  - sketch:    hand-drawn feel (xkcd-style, wavy lines, pencil tones)

Usage:
    from ._render_modes import pick_render_mode, textbook_params, sketch_context

    style = self._random_style()
    mode = pick_render_mode(self._rng)  # clean/textbook/sketch mix
    if mode == "textbook":
        tbp = textbook_params(self._rng)
        fig, ax = plt.subplots(...)
        fig.patch.set_facecolor(tbp["bg"])
        ax.set_facecolor(tbp["bg"])
        # draw with tbp["line_color"], tbp["font_family"], etc.
    elif mode == "sketch":
        with sketch_context():
            fig, ax = plt.subplots(...)
            # draw normally; xkcd context distorts
    else:
        # clean: existing env behavior
        ...
"""
import random
from contextlib import contextmanager
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt

# Default mix: 50% clean, 30% textbook, 20% sketch.
# Can be tuned per-env or globally via env var.
_MODE_WEIGHTS = [("clean", 0.5), ("textbook", 0.3), ("sketch", 0.2)]

def pick_render_mode(rng: random.Random,
                     weights=None) -> str:
    """Pick a render mode using weighted choice."""
    w = weights if weights else _MODE_WEIGHTS
    modes = [m for m, _ in w]
    probs = [p for _, p in w]
    return rng.choices(modes, weights=probs, k=1)[0]

# ------------------------------------------------------------------ #
# Textbook mode
# ------------------------------------------------------------------ #

def textbook_params(rng: random.Random) -> Dict:
    """Style parameters for textbook-style rendering.

    Returns:
        dict with keys:
            bg: background color (white/off-white)
            line_color: primary stroke color (dark grey)
            aux_color: auxiliary/construction line color (light grey)
            fill_color: fill color (very light grey, low alpha)
            font_family: serif font
            number_style: "upright"
            label_style: "italic" (italic for variable letters like a, b, θ)
            line_width: thin (0.6-1.2)
            aux_line_width: thinner (0.4-0.7)
            dpi: 110-130
            figsize_scale: 0.9-1.1
    """
    return {
        "bg": rng.choice([
            "#ffffff", "#fefefe", "#fdfdfd",
            "#fbf8f2", "#faf6ef", "#f8f4ec",   # aged/warm paper
            "#f7f7f4", "#f4f4f0",              # cool paper
        ]),
        "line_color": rng.choice([
            "#111111", "#1a1a1a", "#222222", "#2a2a2a",
            "#0d2233", "#1f2a36",              # near-black with subtle blue
            "#2b1f16",                         # near-black with subtle brown
        ]),
        "aux_color": rng.choice([
            "#888888", "#999999", "#aaaaaa",
            "#7a7a7a", "#b0b0b0", "#6c6c6c",
        ]),
        "fill_color": rng.choice([
            "#d0d0d0", "#c8c8c8", "#bcbcbc", "#b5b5b5",  # visible grey
            "#d8d0c4", "#cfc8bd",                        # warm grey
            "#c8d0d4", "#bfc6cb",                        # cool grey
            "#ddd2bf", "#d2c6b0",                        # beige
        ]),
        "fill_alpha": rng.uniform(0.45, 0.75),
        "font_family": rng.choice(["serif", "DejaVu Serif"]),
        "number_style": "normal",
        "label_style": "italic",
        "line_width": rng.uniform(0.7, 1.2),
        "aux_line_width": rng.uniform(0.4, 0.7),
        "dpi": rng.choice([110, 120, 130]),
        "figsize_scale": rng.uniform(0.9, 1.1),
        "dash_pattern": rng.choice([(4, 2), (3, 2), (5, 3), (2, 1)]),
    }

def apply_textbook_style(fig, ax, tbp: Dict):
    """Apply textbook params to fig/ax (background + spines).

    Individual shape draws still need to use tbp['line_color'], etc.
    """
    fig.patch.set_facecolor(tbp["bg"])
    ax.set_facecolor(tbp["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelsize=9, colors=tbp["line_color"])

def format_label(text: str, tbp: Dict,
                  is_number: bool = False) -> Dict:
    """Return a dict of matplotlib text kwargs for a label in textbook style.

    Numbers stay upright, single-letter variables go italic.
    """
    kwargs = {
        "fontfamily": tbp["font_family"],
        "color": tbp["line_color"],
        "fontsize": 11,
    }
    if not is_number and len(text.strip()) == 1 and text.strip().isalpha():
        kwargs["fontstyle"] = "italic"
    return kwargs

# ------------------------------------------------------------------ #
# Sketch mode (xkcd-style hand-drawn)
# ------------------------------------------------------------------ #

@contextmanager
def sketch_context(scale: float = 1.5, length: float = 100.0,
                    randomness: float = 2.0):
    """Context manager for hand-drawn/sketch rendering.

    Wraps plt.xkcd() but lets caller tune wobble parameters. Use:

        with sketch_context():
            fig, ax = plt.subplots(...)
            # draw
            # fig returned has sketch styling baked in

    Note: sketch mode wants monospace/xkcd-like font. If neither is
    available (as on this cluster), matplotlib warns but still draws.
    """
    with plt.xkcd(scale=scale, length=length, randomness=randomness):
        yield

def sketch_params(rng: random.Random) -> Dict:
    """Style parameters for sketch mode. Muted colors + one accent."""
    return {
        "bg": rng.choice(["#fffdf7", "#fffaf0", "#fdfbf6"]),
        "line_color": rng.choice(["#1a1a1a", "#2a2a2a"]),
        "accent_color": rng.choice([
            "#c0392b", "#27ae60", "#2980b9", "#8e44ad", "#d35400",
        ]),
        "font_family": rng.choice(["Humor Sans", "sans-serif"]),
        "line_width": rng.uniform(1.4, 2.0),
        "dpi": rng.choice([100, 110]),
        "figsize_scale": rng.uniform(0.9, 1.1),
    }

# ------------------------------------------------------------------ #
# Clean mode (pass-through)
# ------------------------------------------------------------------ #

def clean_params(rng: random.Random, base_style: Dict) -> Dict:
    """Clean-mode params are just the existing _random_style() output.

    Provided for symmetry; envs using clean mode can ignore this.
    """
    return dict(base_style)

# ------------------------------------------------------------------ #
# Uniform interface
# ------------------------------------------------------------------ #

def mode_params(mode: str, rng: random.Random,
                 base_style: Dict = None) -> Dict:
    """Return the param dict for a given mode."""
    if mode == "textbook":
        return textbook_params(rng)
    if mode == "sketch":
        return sketch_params(rng)
    return clean_params(rng, base_style or {})
