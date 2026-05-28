"""
Base class for standalone visual QA environments.

These environments generate their own problems (no RLVE text env dependency).
They produce: image + question + short answer, with adaptive difficulty.

Compatible with RLVE_gym_v's manager interface:
  - generate(seed, parameter, timeout_second) -> bool
  - render() -> PIL.Image
  - get_instruction() -> str
  - verify(output) -> dict
  - get_config() -> dict
"""
import math
import os
import re
import random
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
# Disable mpl's automatic dash-scaling-by-linewidth: when an env happens to
# pass linewidth=0 (or near-0) together with a non-solid linestyle, mpl scales
# the dash pattern by lw and ends up at [0, 0], which makes
# GraphicsContextBase.set_dashes raise "At least one value in the dash list
# must be positive". That render exception bubbles out of env.generate(), the
# verifier sees env._answer=None, and a perfectly correct rollout gets graded
# as a failure (~3.5% of val rows on chart-heavy envs). Switching the global
# rcParam off keeps dashes at their literal pattern and removes the trigger.
matplotlib.rcParams['lines.scale_dashes'] = False
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter

def _apply_textbook_filter(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply a textbook-scan-style post-process to a rendered PNG.

    Workflow (each step is random in intensity):
      1. Convert to RGB if not already.
      2. Tint background toward warm paper (#faf4e8).
      3. Desaturate overall color slightly.
      4. Darken / thicken dark lines via brightness drop.
      5. Add very mild gaussian blur for paper-softness.
      6. Slight contrast boost.
      7. Paper-texture overlay (small additive noise pattern).

    Runs entirely in PIL — no extra deps.
    """
    img = img.convert("RGB")
    W, H = img.size

    # 2. Warm-paper background: blend each pixel toward paper color
    #    based on current brightness (dark pixels stay dark).
    paper_colors = [
        (250, 244, 232),   # warm cream
        (245, 240, 225),   # slightly more yellow
        (252, 248, 240),   # lighter
        (248, 243, 233),
        (243, 235, 220),   # aged paper
    ]
    paper_color = rng.choice(paper_colors)
    paper = Image.new("RGB", (W, H), paper_color)
    # Blend ratio: how much to shift background toward paper
    paper_alpha = rng.uniform(0.25, 0.55)
    img = Image.blend(img, paper, paper_alpha)

    # 3. Slight desaturation
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.5, 0.85))

    # 4. Darken lines (reduce brightness), but only slightly
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.0))

    # 5. Mild blur
    if rng.random() < 0.65:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.9)))

    # 6. Contrast up slightly
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(1.05, 1.25))

    # 7. Add very mild noise for paper texture
    if rng.random() < 0.7:
        import numpy as np
        arr = np.array(img).astype(np.int16)
        noise = np.random.RandomState(rng.randint(0, 1 << 30)).randint(
            -8, 8, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        img = Image.fromarray(arr, "RGB")

    return img

def _reset_mpl_state():
    """Reset matplotlib state to recover from intermittent render errors.

    Closes all open figures and reloads rcParamsDefault, then re-applies our
    own rcParam overrides (chiefly `lines.scale_dashes=False` — see module
    header for why).
    """
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
        import matplotlib as _mpl
        _mpl.rcParams.update(_mpl.rcParamsDefault)
        _mpl.rcParams['lines.scale_dashes'] = False
    except Exception:
        pass


def _verify_diag_log(env_name, seed, parameter, predicted, gt, correct):
    """Append every verify call to disk for debugging. Off by default."""
    try:
        import os, time, json
        log_path = os.environ.get(
            "STANDALONE_VERIFY_DIAG_LOG",
            os.path.join(os.environ.get("BENCHMARK_STATE_DIR", "/tmp"), "verify_diag.jsonl")
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "env": env_name,
            "seed": seed,
            "parameter": parameter,
            "predicted": (predicted or "")[:120],
            "gt": (gt or "")[:120],
            "correct": bool(correct),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _gen_diag_log(env_name, seed, parameter, reason, detail):
    """Append generate-failure diagnostics to disk so failures aren't silent."""
    try:
        import os, time, json
        log_path = os.environ.get(
            "STANDALONE_GEN_DIAG_LOG",
            os.path.join(os.environ.get("BENCHMARK_STATE_DIR", "/tmp"), "gen_failures.jsonl")
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "env": env_name,
            "seed": seed,
            "parameter": parameter,
            "reason": reason,
        }
        if detail:
            rec["detail"] = detail
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


class StandaloneVisualEnv(ABC):
    """Base for environments that generate their own data + image + answer."""

    ENV_NAME: str = None

    # 2026-05-05: Image augmentation switches per env. Override to False in
    # envs whose answer depends on image orientation (transformation,
    # protractor, mirror, compass, bearing, rotation_*).
    ALLOW_ROTATION = True   # ±2° random rotation at 15% probability
    ALLOW_NOISE = True      # JPEG/blur/gauss/bg-color (~30% combined chance)

    # Per-call deterministic wrapper assignment (2026-05-04 user direction):
    # don't say "any of these wrappers" — model picks unpredictably. Instead,
    # tell the model EXACTLY which wrapper to use for this problem (varied
    # across seeds so training data sees all 3 wrapper styles).
    _WRAPPER_INSTRUCTIONS = [
        "Think step by step. Place your final answer inside <answer>X</answer> at the end.",
        "Think step by step. Provide your final answer in the form \\boxed{X} at the end.",
        "Think step by step. End your response with `Final answer: X` on the last line.",
    ]
    # Default template (kept for back-compat; per-call wrapping happens in
    # get_full_instruction below).
    REASONING_TEMPLATE = (
        "{instruction}\n\n"
        "Think step by step, then give your final answer in any of these "
        "wrappers: <answer>X</answer>, \\boxed{{X}}, or `Final answer: X`."
    )

    def __init__(self):
        self.seed: Optional[int] = None
        self.parameter: Optional[Dict] = None
        self._rng: Optional[random.Random] = None
        self._answer: Optional[str] = None
        self._question: Optional[str] = None
        self._image: Optional[Image.Image] = None
        # Per-instance shape config (defaults from env vars so the launcher
        # can switch the whole training run to shape mode with one flag).
        # Subclasses may override after super().__init__() to opt out or tune.
        import os as _os
        self.shape_strategy = _os.environ.get("RLVE_SHAPE_STRATEGY",
                                              type(self).shape_strategy)
        try:
            self.shape_beta = float(_os.environ.get("RLVE_SHAPE_BETA",
                                                    type(self).shape_beta))
        except (ValueError, TypeError):
            self.shape_beta = type(self).shape_beta

    # ------------------------------------------------------------------ #
    # Core interface (compatible with RLVE_gym_v manager)
    # ------------------------------------------------------------------ #

    def generate(self, seed: int, parameter: Dict, timeout_second: int = 10) -> bool:
        """Generate a problem. Returns True on success.

        Internally retries up to 3 times because matplotlib has intermittent
        errors under high reward-eval concurrency (e.g. "dash list must be
        positive" thrown by GraphicsContext.set_dashes when a font / dpi /
        backend state combination produces an all-zero dash array). Each retry
        first resets matplotlib pyplot state (close all figures + reload
        rcParamsDefault) so the next attempt sees a clean slate.
        """
        self.seed = seed
        self.parameter = parameter
        for attempt in range(3):
            # Re-seed each attempt so retries use the same problem distribution.
            # _generate_problem itself iterates internally for shape constraints;
            # this outer retry recovers from matplotlib state corruption.
            self._rng = random.Random(seed)
            try:
                result = self._generate_problem(seed, parameter)
                if result is None:
                    if attempt == 2:
                        _gen_diag_log(self.ENV_NAME, seed, parameter, "returned_none", None)
                        return False
                    # Reset mpl state and try again
                    _reset_mpl_state()
                    continue
                self._question, self._answer, self._image = result
                return True
            except Exception as e:
                import traceback
                last_tb = traceback.format_exc()[:500]
                last_exc = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt == 2:
                    _gen_diag_log(self.ENV_NAME, seed, parameter, "exception",
                                  f"{last_exc}\n{last_tb}")
                    return False
                # Log intermediate failure too so we can see flake rate
                _gen_diag_log(self.ENV_NAME, seed, parameter,
                              f"exception_attempt_{attempt}", last_exc)
                _reset_mpl_state()
        return False

    def render(self) -> Image.Image:
        assert self._image is not None, "Call generate() first"
        img = self._image
        # 2026-05-05: opt-in image augmentation — only applied when
        # RLVE_AUGMENT=1 env var is set (training time, not eval).
        if os.environ.get("RLVE_AUGMENT") == "1":
            img = self._apply_augmentation(img)
        # 2026-05-07: cap image size to bound Qwen2.5-VL image_grid_thw tokens.
        # BISECT TEST: re-enabled to verify slowdown reproducible.
        max_pixels = int(os.environ.get("RLVE_MAX_PIXELS", "0"))
        if max_pixels > 0:
            w, h = img.size
            if w * h > max_pixels:
                scale = (max_pixels / (w * h)) ** 0.5
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def _apply_augmentation(self, img: Image.Image) -> Image.Image:
        """Probability-triggered low-risk augmentation for training robustness.
        2026-05-05 user direction: 70% pass-through, 30% trigger ONE
        randomly-chosen augmentation type. Skipped at eval (no RLVE_AUGMENT).
        Settings tuned per user feedback (V6+V7).

        Size-jitter step (always-on at training): real benchmarks have
        varied image dimensions while ~16% of our envs render at fixed
        sizes. Pad each side 0-40px white (seed-deterministic) so every
        training image has a unique size — matches benchmark distribution
        and improves resolution robustness.
        """
        import io as _io
        import random as _random
        rng = _random.Random((self.seed or 0) * 7919 + 13)
        # ALWAYS: tiny size jitter (independent of 70/30 trigger). Adds
        # 0-40px white pad on each side — every training image gets a
        # unique size with no content distortion.
        pl = rng.randint(0, 40)
        pt = rng.randint(0, 40)
        pr = rng.randint(0, 40)
        pb = rng.randint(0, 40)
        if pl + pt + pr + pb > 0:
            try:
                w, h = img.size
                jittered = Image.new("RGB", (w + pl + pr, h + pt + pb),
                                     (255, 255, 255))
                jittered.paste(img.convert("RGB"), (pl, pt))
                img = jittered
            except Exception:
                pass
        # 70% pass-through (skip remaining augmentations)
        if rng.random() >= 0.30:
            return img
        # 30% pick ONE augmentation
        choices = ["jpeg", "bg_tint", "blur", "gauss_noise"]
        if self.ALLOW_ROTATION:
            choices.append("rotate")
        if not self.ALLOW_NOISE:
            return img
        op = rng.choice(choices)
        try:
            if op == "rotate":
                # ±3-8° random direction; pad first to keep content complete
                deg = rng.uniform(3, 8) * rng.choice([-1, 1])
                w, h = img.size
                pad = max(int(w * 0.20), int(h * 0.20))
                padded = Image.new("RGB", (w + 2*pad, h + 2*pad),
                                    (255, 255, 255))
                padded.paste(img.convert("RGB"), (pad, pad))
                img = padded.rotate(deg, resample=Image.BILINEAR,
                                    expand=False, fillcolor=(255, 255, 255))
            elif op == "jpeg":
                q = rng.randint(8, 20)
                buf = _io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=q)
                buf.seek(0)
                img = Image.open(buf).copy()
            elif op == "bg_tint":
                shift = rng.choice([-40, 40])
                img = Image.eval(img.convert("RGB"),
                                 lambda v: max(0, min(255, v + shift)))
            elif op == "blur":
                sigma = rng.uniform(1.0, 3.0)
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            elif op == "gauss_noise":
                import numpy as _np
                noise_amp = rng.randint(30, 100)
                arr = _np.array(img.convert("RGB"), dtype=_np.int16)
                noise = _np.random.randint(-noise_amp, noise_amp + 1,
                                           size=arr.shape, dtype=_np.int16)
                arr = _np.clip(arr + noise, 0, 255).astype(_np.uint8)
                img = Image.fromarray(arr)
        except Exception:
            return self._image
        return img

    def get_instruction(self) -> str:
        assert self._question is not None, "Call generate() first"
        return self._question

    def get_full_instruction(self) -> str:
        # Per-call deterministic wrapper assignment (varied by seed so training
        # sees all 3 wrapper styles, but each individual call has ONE clear
        # instruction). Verifier still accepts all 3 wrappers regardless.
        idx = (self.seed or 0) % len(self._WRAPPER_INSTRUCTIONS)
        wrapper_text = self._WRAPPER_INSTRUCTIONS[idx]
        return f"{self.get_instruction()}\n\n{wrapper_text}"

    def get_prompt(self) -> Tuple[Image.Image, str]:
        return self.render(), self.get_full_instruction()

    def get_config(self) -> Dict:
        return {
            "seed": self.seed,
            "parameter": self.parameter,
            "passing_reward_threshold": 0.5,
        }

    def set_config(self, config: Dict):
        """Restore from config (used by reward function when generate() fails)."""
        self.seed = config.get("seed")
        self.parameter = config.get("parameter")
        if self.seed is not None and self.parameter is not None:
            self.generate(self.seed, self.parameter)

    # text_env compatibility — reward_function.py falls back to env.text_env.set_config()
    @property
    def text_env(self):
        return self  # self acts as its own text_env

    # v6 shape reward configuration. Subclasses can override these to opt into
    # continuous reward shaping for wrong answers. Default behaviour stays
    # binary (reward=-0.5 on wrong) so existing envs are unchanged.
    #
    #   shape_strategy ∈ {"binary", "min_over_max", "list_mean", "auto"}
    #     binary       — wrong → -0.5 (legacy v5 behaviour)
    #     min_over_max — both pred/gt parse as float ⇒ (min/max)^β reward
    #                    (RLVE 'sum_triangle_area' style); else binary fallback.
    #     list_mean    — both parse as comma-separated lists same length ⇒
    #                    (correct_count / N)^β (RLVE 'random_range' style);
    #                    else binary fallback.
    #     auto         — try min_over_max, then list_mean, else binary.
    #   shape_beta     — RLVE-style exponent. Larger = stricter (only near-
    #                    exact gets credit). 0.5 = gentle, 2 = strict, 10 = ~binary.
    #   wrong_format_score / invalid_solution_score — penalty floors when
    #                    parsing fails. Defaults match RLVE.
    shape_strategy: str = "binary"
    shape_beta: float = 2.0
    wrong_format_score: float = -1.0
    invalid_solution_score: float = -1.0

    # A11 (2026-05-03): per-benchmark numeric tolerance overrides. Subclasses
    # can set these class variables to tighten the numeric branch of
    # _check_answer when the target benchmark scorer requires stricter
    # tolerance than the default (5% relative OR 0.5 absolute).
    #
    #   BENCHMARK_NUM_TOLERANCE_ABS — when not None, numeric branch uses
    #     abs(p - g) <= TOL instead of the default 5%/0.5 path.
    #     Use for tight-precision (0.001) numeric scorers.
    #   BENCHMARK_NUM_TOLERANCE_REL — when not None and ABS is None, numeric
    #     branch uses abs(p - g) / max(abs(g), 1) <= TOL instead of default.
    #     Use for 1%-relative-tolerance numeric scorers.
    #   When both are None, the legacy 5%-rel-or-0.5-abs path is used (no
    #     change to existing envs).
    BENCHMARK_NUM_TOLERANCE_ABS: Optional[float] = None
    BENCHMARK_NUM_TOLERANCE_REL: Optional[float] = None

    def _score_continuous(self, predicted: str, gt: str) -> Optional[float]:
        """Return shaped reward in [-1, 1] for wrong answers. None ⇒ caller
        falls back to binary -0.5. Right answers are not routed through this
        (verify gives them 1.0 directly)."""
        strategy = self.shape_strategy
        if strategy == "binary":
            return None

        # In "auto" mode, dispatch list_mean first when both sides have
        # multiple comma-separated tokens — otherwise the min_over_max branch
        # would happily strip the commas (legacy thousands handling) and
        # treat e.g. "9,9,9,9,9" as the integer 99999, scoring it against a
        # different gold-as-int and giving a misleading ratio.
        if strategy in ("list_mean", "auto"):
            try:
                if isinstance(predicted, str) and isinstance(gt, str) \
                        and "," in predicted and "," in gt:
                    pred_list = [x.strip() for x in predicted.split(",")]
                    gt_list = [x.strip() for x in gt.split(",")]
                    if len(gt_list) >= 2:
                        if len(pred_list) != len(gt_list):
                            return self.invalid_solution_score
                        correct = sum(p == g for p, g in zip(pred_list, gt_list))
                        return (correct / len(gt_list)) ** self.shape_beta
            except (AttributeError, TypeError):
                pass

        if strategy in ("min_over_max", "auto"):
            try:
                # Only strip a single trailing %; do NOT strip commas (those
                # are list separators, not thousands).
                p_str = predicted[:-1] if predicted.endswith("%") else predicted
                g_str = gt[:-1] if gt.endswith("%") else gt
                # Reject if the string contains a comma — that's a list, not
                # a number with thousands separators (we handle thousands by
                # explicit re-strip only when the rest is all digits).
                def _to_float(s):
                    if "," in s:
                        # try thousands-separator pattern: digits in 1-3 digit
                        # groups separated by commas, optional minus.
                        if re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", s):
                            return float(s.replace(",", ""))
                        raise ValueError
                    return float(s)
                p = _to_float(p_str)
                g = _to_float(g_str)
                if abs(g) < 1e-9:
                    return 1.0 if abs(p) < 1e-9 else self.invalid_solution_score
                if (p < 0) != (g < 0):
                    return self.invalid_solution_score
                ratio = min(abs(p), abs(g)) / max(abs(p), abs(g))
                return ratio ** self.shape_beta
            except (ValueError, TypeError, AttributeError):
                pass

        return None

    def verify(self, output: str) -> Dict[str, Union[float, int]]:
        """Verify model output. Returns dict with reward, accuracy, format_score
        (matching VisualVerifiableEnvironment's interface)."""
        # Defensive: if generate() failed silently, _answer is None. Returning
        # -0.5 here would punish a correct answer that we just can't verify;
        # instead emit a neutral marker so compute_score can log + skip.
        if self._answer is None:
            return {"reward": 0.0, "format_score": 0, "accuracy": 0,
                    "env_unverifiable": 1}

        # 2026-05-04 STRICT FORMAT: only accept the wrapper that matches the
        # one this seed's prompt asked for. Was previously permissive (any of
        # 3 wrappers). User caught the bug: "prompt asks for \boxed but model
        # uses <answer> and verify still passes" — model never learns to obey
        # format instruction. a logic benchmark infographic -20.83 was caused by
        # this kind of format mismatch in the opposite direction.
        wrapper_idx = (self.seed or 0) % len(self._WRAPPER_INSTRUCTIONS)
        # Define each wrapper's extractor as a lambda over `output` that
        # returns a match-like obj with .group(1) → predicted_raw, or None.
        def _extract_answer_tag(out: str):
            m = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
            return m
        def _extract_boxed(out: str):
            bm = re.search(r"\\boxed\{", out)
            if not bm:
                return None
            start = bm.end()
            depth = 1
            i = start
            while i < len(out) and depth > 0:
                if out[i] == "{":
                    depth += 1
                elif out[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                raw = out[start:i - 1]
                return type("M", (), {"group": lambda self_, n=1: raw})()
            return None
        def _extract_final_answer(out: str):
            # 2026-05-04 audit fix: the previous regex used `\.` as a
            # terminator, which truncated decimal answers like "0.125" → "0".
            # Removed `\.`; trailing period is handled by _check_answer.rstrip.
            return re.search(
                r"(?i)final\s+answer\s*[:：]?\s*\*?\*?\s*(.+?)(?:\n|$|\*\*)",
                out, re.DOTALL,
            )
        # Wrapper idx ↔ extractor mapping must match _WRAPPER_INSTRUCTIONS order.
        _extractors = [_extract_answer_tag, _extract_boxed, _extract_final_answer]
        answer_match = _extractors[wrapper_idx](output)
        if not answer_match:
            # 2026-05-04 STRICT FORMAT: if model used a DIFFERENT wrapper
            # than the one prompt requested, fail format. Don't fall back to
            # bare-answer extraction. This forces the model to obey the
            # specific format instruction.
            other_wrappers_present = any(
                _extractors[i](output) is not None
                for i in range(len(_extractors)) if i != wrapper_idx
            )
            if other_wrappers_present:
                return {"reward": self.wrong_format_score,
                        "format_score": 0, "accuracy": 0}
            # If no wrapper at all, allow bare-answer fallback.
            stripped = output.strip()

            def _looks_like_prose(s: str) -> bool:
                lower = s.lower()
                return (
                    ". " in s or ", " in s[:max(1, len(s)-10)]
                    or " is " in lower or " the " in lower
                    or " maybe " in lower or " think " in lower
                    or s.count(".") >= 2
                )

            if 0 < len(stripped) <= 100 and "\n" not in stripped:
                if not _looks_like_prose(stripped):
                    gt = str(self._answer).strip()
                    if self._check_answer(stripped, gt):
                        return {"reward": 1.0, "format_score": 1, "accuracy": 1}

            # A11 (2026-05-03): NEW last-line extraction for CoT-style outputs.
            # When the model generates reasoning followed by a bare answer on
            # the final line (e.g., bare-text Multi-Choice / Fact-Checking
            # eval style: "...analyzing the chart...\n\nItaly"), extract the
            # last non-empty line and try it as the bare answer (subject to
            # the same prose check). This is gated on multi-line outputs
            # only — single-line goes through the bare-output branch above.
            if "\n" in stripped and 0 < len(stripped) <= 5000:
                # Find last non-empty line
                last_line = ""
                for line in reversed(stripped.split("\n")):
                    candidate = line.strip()
                    if candidate:
                        last_line = candidate
                        break
                if 0 < len(last_line) <= 100 and not _looks_like_prose(last_line):
                    gt = str(self._answer).strip()
                    if self._check_answer(last_line, gt):
                        return {"reward": 1.0, "format_score": 1, "accuracy": 1}

            return {"reward": self.wrong_format_score,
                    "format_score": 0, "accuracy": 0}

        predicted = answer_match.group(1).strip()
        gt = str(self._answer).strip()

        correct = self._check_answer(predicted, gt)
        # Diag: log every verify call (cheap; off unless STANDALONE_VERIFY_DIAG=1)
        import os as _os
        if _os.environ.get("STANDALONE_VERIFY_DIAG", "0") == "1":
            _verify_diag_log(self.ENV_NAME, self.seed, self.parameter,
                             predicted, gt, correct)
        if correct:
            reward = 1.0
        else:
            shaped = self._score_continuous(predicted, gt)
            reward = -0.5 if shaped is None else float(shaped)
            # Clamp into [-1, 1] just in case.
            reward = max(-1.0, min(1.0, reward))
        return {
            "reward": reward,
            "format_score": 1,
            "accuracy": 1 if correct else 0,
        }

    # ------------------------------------------------------------------ #
    # Answer checking
    # ------------------------------------------------------------------ #

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.
        Handles numeric (with tolerance), symbolic math, and string matching."""
        # Clean
        predicted = predicted.strip().lower().rstrip(".")
        ground_truth = ground_truth.strip().lower().rstrip(".")

        # Strip degree symbols, units
        predicted = predicted.replace("°", "").replace("degrees", "").strip()
        ground_truth = ground_truth.replace("°", "").replace("degrees", "").strip()

        # Exact match
        if predicted == ground_truth:
            return True

        # --- MCQ letter variant handling ---
        # Handle "A)", "(A)", "[A]", "The answer is A", "A is correct", etc.
        if len(ground_truth) == 1 and ground_truth.upper() in "ABCDEFGH":
            # Try patterns like "(A)", "A)", "[A]" first (explicit MCQ markers)
            mcq_match2 = re.search(r'[\(\[]\s*([a-hA-H])\s*[\)\]]', predicted)
            if mcq_match2:
                return mcq_match2.group(1).upper() == ground_truth.upper()
            # Find first standalone letter in prediction
            mcq_match = re.search(r'\b([a-hA-H])\b', predicted)
            if mcq_match:
                return mcq_match.group(1).upper() == ground_truth.upper()

        # --- Numeric with unit handling ---
        # Try to extract first number from prediction (handles "5 units", "x = 5", etc.)
        num_match = re.search(r'[-+]?\d*\.?\d+', predicted)
        if num_match:
            try:
                p_num = float(num_match.group())
                g_num = float(ground_truth.replace(",", "").replace("%", ""))
                if self._numeric_match(p_num, g_num):
                    return True
            except (ValueError, TypeError):
                pass

        # --- Yes/No narrative handling ---
        if ground_truth in ["yes", "no"]:
            # Look for yes/no as first meaningful word
            first_word_match = re.match(r'^\s*(yes|no|y|n)\b', predicted)
            if first_word_match:
                w = first_word_match.group(1)
                return (w.startswith("y") and ground_truth == "yes") or (w.startswith("n") and ground_truth == "no")

        # Try evaluating both as math expressions, then compare numerically
        p_val = self._try_eval_expr(predicted)
        g_val = self._try_eval_expr(ground_truth)

        if p_val is not None and g_val is not None:
            if self._numeric_match(p_val, g_val):
                return True

        # Ratio matching: "2:2" == "1:1", "3 : 1" == "3:1"
        p_ratio = self._parse_ratio(predicted)
        g_ratio = self._parse_ratio(ground_truth)
        if p_ratio is not None and g_ratio is not None:
            if abs(p_ratio - g_ratio) < 0.01:
                return True

        # Check if ground truth is contained in prediction (for single-word answers)
        if len(ground_truth) >= 3 and ground_truth in predicted:
            # Only if prediction doesn't contain too much extra
            if len(predicted) < len(ground_truth) * 3:
                return True

        return False

    def _numeric_match(self, p_num: float, g_num: float) -> bool:
        """Compare two numbers using benchmark-specific tolerance overrides.

        Default (both class vars None): 5% relative OR 0.5 absolute (legacy).
        BENCHMARK_NUM_TOLERANCE_ABS != None: abs(p - g) <= TOL (tight-precision 0.001).
        BENCHMARK_NUM_TOLERANCE_REL != None: abs(p - g) / max(abs(g), 1) <= TOL
            (1% relative 0.01) — relative-only, no absolute floor.

        ABS takes precedence over REL when both are set.
        """
        try:
            diff = abs(p_num - g_num)
        except (TypeError, ValueError):
            return False
        abs_tol = type(self).BENCHMARK_NUM_TOLERANCE_ABS
        rel_tol = type(self).BENCHMARK_NUM_TOLERANCE_REL
        # Allow per-instance override too (rarely used, but safe)
        if abs_tol is None:
            abs_tol = getattr(self, "BENCHMARK_NUM_TOLERANCE_ABS", None)
        if rel_tol is None:
            rel_tol = getattr(self, "BENCHMARK_NUM_TOLERANCE_REL", None)
        if abs_tol is not None:
            return diff <= abs_tol
        if rel_tol is not None:
            return diff / max(abs(g_num), 1.0) <= rel_tol
        # Legacy default: 5% relative OR 0.5 absolute
        if abs(g_num) < 1e-9:
            return abs(p_num) < 0.01
        return diff / max(abs(g_num), 1e-9) < 0.05 or diff < 0.5

    @staticmethod
    def _parse_ratio(s: str) -> float:
        """Try to parse a ratio like '2:3' or '1 : 1' into a float."""
        m = re.match(r'^\s*(\d+)\s*:\s*(\d+)\s*$', s)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den > 0:
                return num / den
        return None

    @staticmethod
    def _try_eval_expr(s: str) -> float:
        """Try to evaluate a string as a math expression. Returns float or None.

        Handles symbolic expressions involving pi, sqrt, ², ³, fractions, and
        common LaTeX (\\pi, ^, \\frac, \\cdot). Implicit multiplication
        between digit/letter and pi/sqrt is inserted automatically so that
        outputs like "25π", "16π + 54", "2√37" evaluate correctly.
        """
        if not s:
            return None
        # Reject comma-separated lists ("1,2,3,4") so we don't accidentally
        # collapse them into the integer 1234 — that would make the numeric
        # tolerance branch falsely match "1,2,9,4,5" ≈ "1,2,3,4,5".
        # Only allow commas if they form a thousands-separator pattern.
        def _strip_thousands_commas(x):
            if "," not in x:
                return x
            if re.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$", x):
                return x.replace(",", "")
            raise ValueError("comma-list, not a number")
        # Direct float
        try:
            return float(_strip_thousands_commas(s).replace("%", ""))
        except ValueError:
            pass
        try:
            expr = _strip_thousands_commas(s)
            # Strip leading "X = " variable assignments
            expr = re.sub(r'^\s*[A-Za-z][A-Za-z0-9_]*\s*=\s*', '', expr)
            # LaTeX → simpler forms first
            expr = expr.replace("\\pi", "π")
            expr = expr.replace("\\cdot", "*").replace("\\times", "*")
            expr = expr.replace("^", "**")
            expr = expr.replace("²", "**2").replace("³", "**3")
            # Handle sqrt(N) → math.sqrt(N) — do this FIRST before √ substitution
            expr = re.sub(r'(?<!math\.)sqrt\(([^)]+)\)', r'math.sqrt(\1)', expr)
            # Handle N√M → N*math.sqrt(M), e.g. "2√37" → "2*math.sqrt(37)"
            expr = re.sub(r'(\d+)√(\d+)', r'\1*math.sqrt(\2)', expr)
            # Handle standalone √N → math.sqrt(N)
            expr = re.sub(r'√(\d+)', r'math.sqrt(\1)', expr)
            # Insert implicit multiplication around π / pi
            # 25π → 25*π, π25 → π*25
            expr = re.sub(r'(\d+\.?\d*)\s*π', r'\1*π', expr)
            expr = re.sub(r'π\s*(\d+\.?\d*)', r'π*\1', expr)
            expr = re.sub(r'(\d+\.?\d*)\s*\bpi\b', r'\1*pi', expr)
            expr = re.sub(r'\bpi\b\s*(\d+\.?\d*)', r'pi*\1', expr)
            # implicit "(...)" mult: e.g. "2(3)" → "2*(3)"
            expr = re.sub(r'(\d+\.?\d*)\s*\(', r'\1*(', expr)
            # \frac{a}{b} → ((a)/(b))
            expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', expr)
            expr = re.sub(r'\\sqrt\{([^{}]+)\}', r'math.sqrt(\1)', expr)
            # Now substitute pi numeric
            expr = expr.replace("π", str(math.pi))
            expr = re.sub(r'\bpi\b', str(math.pi), expr)
            # Safety: only allow digits, math ops, math.sqrt, parens, dots, spaces
            cleaned = expr.replace('math.sqrt', '').replace(str(math.pi), '')
            if re.match(r'^[\d\s\+\-\*/\.\(\)]+$', cleaned):
                return float(eval(expr, {"__builtins__": {}, "math": math}))
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Abstract: subclasses implement these
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        """Generate a single problem instance.

        Returns:
            (question_text, answer_string, rendered_image) or None if generation fails.
        """
        pass

    # ------------------------------------------------------------------ #
    # Visual style randomization
    # ------------------------------------------------------------------ #

    # 8 color palettes — diverse visual styles
    _COLOR_PALETTES = [
        ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"],
        ["#4285f4", "#ea4335", "#fbbc05", "#34a853", "#ff6d01", "#46bdc6", "#7b1fa2", "#c2185b"],
        ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#606c38", "#283618", "#dda15e"],
        ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600", "#374c80", "#7a5195", "#ef5675"],
        ["#1d3557", "#457b9d", "#a8dadc", "#e63946", "#f1faee", "#2b2d42", "#8d99ae", "#ef233c"],
        # Dark theme
        ["#bb86fc", "#03dac6", "#cf6679", "#ffb74d", "#81c784", "#64b5f6", "#f06292", "#a1887f"],
        # Pastel
        ["#ffadad", "#ffd6a5", "#fdffb6", "#caffbf", "#9bf6ff", "#a0c4ff", "#bdb2ff", "#ffc6ff"],
        # High contrast
        ["#000000", "#e60000", "#0000e6", "#00b300", "#e6e600", "#e600e6", "#00e6e6", "#ff8000"],
    ]
    _BG_COLORS = ["#ffffff", "#f8f9fa", "#fefefe", "#f5f5f5", "#fff8f0",
                   "#f0f4f8", "#fdf2f8", "#f0fdf4", "#fef3c7"]
    _FONT_FAMILIES = ["DejaVu Sans", "serif", "sans-serif"]

    def _random_style(self) -> Dict:
        """Return a dict of randomized visual style parameters.
        Includes chart layout, geometry line style, and diagram node shape variants."""
        rng = self._rng
        palette_idx = rng.randint(0, len(self._COLOR_PALETTES) - 1)
        theme = rng.choice(["default", "minimal", "bold", "scientific"])
        return {
            "bg_color": rng.choice(self._BG_COLORS),
            "palette": self._COLOR_PALETTES[palette_idx],
            "font_family": rng.choice(self._FONT_FAMILIES),
            "font_size_base": rng.choice([9, 10, 11, 12, 13, 14]),
            "line_width": rng.choice([1.0, 1.5, 2.0, 2.5]),
            "show_grid": rng.choice([True, False]),
            "grid_alpha": rng.uniform(0.08, 0.4),
            "grid_style": rng.choice(["--", "-.", ":"]),
            "dpi": rng.choice([100, 110, 120]),
            "spine_top": rng.choice([True, False]),
            "spine_right": rng.choice([True, False]),
            "theme": theme,
            "legend_loc": rng.choice(["best", "upper right", "upper left", "lower right"]),
            "geo_line_color": rng.choice(["#2c3e50", "#1a5276", "#7b241c", "#0d6efd", "#198754"]),
            "geo_fill_alpha": rng.uniform(0.15, 0.45),
            "label_box": rng.choice([True, False]),
            "node_shape": rng.choice(["circle", "circle", "rounded_rect"]),
            "edge_style": rng.choice(["straight", "straight", "curved"]),
            "figsize_scale": rng.uniform(0.85, 1.15),
        }

    def _apply_style(self, fig, ax, style: Dict):
        """Apply randomized style to a matplotlib figure/axes."""
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        if style.get("show_grid"):
            ax.grid(True, alpha=style.get("grid_alpha", 0.2),
                   linestyle=style.get("grid_style", "--"))
        ax.spines["top"].set_visible(style.get("spine_top", False))
        ax.spines["right"].set_visible(style.get("spine_right", False))
        theme = style.get("theme", "default")
        if theme == "bold":
            for spine in ax.spines.values():
                spine.set_linewidth(2)
        elif theme == "minimal":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        elif theme == "scientific":
            ax.tick_params(direction="in")

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    # v4 B1 retrofit: any env can opt into textbook-style postprocessing by
    # setting this class flag. When True, `fig_to_pil` randomly (probability
    # ~25% seeded by self._rng) applies a textbook-style filter to the
    # rendered PNG — warm paper background, darkened ink, mild edge wobble.
    # Purpose: close the style gap between synthetic env renders and the
    # textbook-scanned figures in targeted-geometry / targeted-vision-math / word-math.
    TEXTBOOK_POSTPROCESS: bool = False
    _TEXTBOOK_RATE: float = 0.30

    def fig_to_pil(self, fig, dpi=120) -> Image.Image:
        """Convert matplotlib figure to PIL Image. Optionally apply
        textbook-style post-processing when TEXTBOOK_POSTPROCESS is set
        on the subclass.
        """
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor(), dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()

        if self.TEXTBOOK_POSTPROCESS and self._rng is not None:
            # RNG seeded by env seed so the same problem renders
            # consistently across reward re-rolls.
            if self._rng.random() < self._TEXTBOOK_RATE:
                img = _apply_textbook_filter(img, self._rng)
        return img
