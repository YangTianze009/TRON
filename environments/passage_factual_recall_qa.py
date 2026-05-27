"""
Passage Factual Recall QA — reference L128/L129-style short factual MCQ
rendered as text-in-image.

Style: PURE-OCR pages — no diagrams. The "image" is a short factual
passage (1-3 sentences) plus a multiple-choice question with 4 options
A/B/C/D. The model OCRs the passage and the question, then picks the
correct option letter.

Sample target style (from design notes):
  *idx=180* — "What does LED stand for? (A)low energy display
              (B)light emitting display (C)light emitting diode
              (D)light emitting detector." Ans: C.
  *idx=181* — "The function of a fuse is to: (A)reduce electricity use
              (B)produce by windpower (C)provide overcurrent protection
              (D)none." Ans: C.

Includes the L129 careful-converse subtemplate ("flag colors of Slavic
countries are red/blue/white. The UK has red/blue/white in its flag.
The UK is a Slavic country. (A)T (B)F (C)Insufficient.") — with a small
parameterized variant family.

Single-letter MCQ answer (A/B/C/D).
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


# ===================================================================== #
# Factual recall question bank
# Each item: dict with
#   passage: short context paragraph (or None if question is self-contained)
#   stem:    the question itself
#   options: list of 4 strings (correct first; will be shuffled)
#   topic:   tag (electronics, physics, mechanics, biology, chemistry, ...)
# ===================================================================== #

_FACTUAL_BANK: List[Dict] = [
    # Electronics / EE
    {
        "passage": "An LED is a semiconductor light source that emits light when current flows through it.",
        "stem": "What does LED stand for?",
        "options": ["light emitting diode", "low energy display", "light emitting display", "light emitting detector"],
        "topic": "electronics",
    },
    {
        "passage": "A fuse is a safety device installed in series with an electrical circuit.",
        "stem": "The function of a fuse is to:",
        "options": ["provide overcurrent protection", "reduce electricity use", "produce power by wind", "store electrical energy"],
        "topic": "electronics",
    },
    {
        "passage": "A capacitor is a passive two-terminal electrical component.",
        "stem": "A device used to store electrical energy is called a:",
        "options": ["Capacitor", "Resistor", "Diode", "Inductor"],
        "topic": "electronics",
    },
    {
        "passage": "A resistor is a passive electrical component used to oppose the flow of current.",
        "stem": "Which component is used to limit current in a circuit?",
        "options": ["Resistor", "Capacitor", "Diode", "Inductor"],
        "topic": "electronics",
    },
    {
        "passage": "A diode allows electric current to flow primarily in one direction.",
        "stem": "Which component allows current to flow in only one direction?",
        "options": ["Diode", "Resistor", "Capacitor", "Inductor"],
        "topic": "electronics",
    },
    {
        "passage": "Internal combustion engines run through several cycles per stroke.",
        "stem": "There are four cycles in IC engines: compression, exhaust, intake, expansion. In a two-stroke engine, two functions occur in one stroke. Which two stages occur simultaneously?",
        "options": ["Compression and intake", "Compression and exhaust", "Intake and expansion", "Exhaust and expansion"],
        "topic": "mechanical",
    },
    {
        "passage": "A microprocessor is a single integrated circuit that contains the data-processing logic of a computer.",
        "stem": "Which device executes logical commands to control sensors and actuators?",
        "options": ["I/O module", "Processor", "RAM", "ROM"],
        "topic": "electronics",
    },
    # Refrigeration / heat
    {
        "passage": "A refrigerator is a cooling appliance with cooling coils whose effectiveness depends on heat transfer.",
        "stem": "The formation of frost on the cooling coils in a refrigerator:",
        "options": ["Reduces power efficiency", "Improves performance", "Reduces heat transfer", "Has no effect"],
        "topic": "thermal",
    },
    {
        "passage": "Heat moves naturally from hot regions toward cold regions.",
        "stem": "What is the main mechanism by which a hot cup of coffee cools down in a cool room?",
        "options": ["Heat transfer to the surroundings", "Internal pressure decrease", "Light emission", "Magnetic resonance"],
        "topic": "thermal",
    },
    # Buoyancy / floating
    {
        "passage": "An object floats in water if its average density is less than water.",
        "stem": "Which of the following objects will NOT float on water?",
        "options": ["A pair of metal scissors", "A banana", "An empty plastic soda bottle", "A wooden pencil"],
        "topic": "physics",
    },
    {
        "passage": "Density determines whether an object will float or sink in a fluid.",
        "stem": "An object will sink in water when:",
        "options": ["its density is greater than that of water", "its mass is greater than 1 kg", "it has sharp edges", "it is shiny"],
        "topic": "physics",
    },
    # Vehicle dynamics
    {
        "passage": "The acceleration of a vehicle depends on multiple physical factors.",
        "stem": "Which of the following might directly affect the rate of acceleration of a vehicle?",
        "options": ["All of the above", "Total weight", "Engine power", "Drag coefficient"],
        "topic": "mechanical",
    },
    # Forces / mechanics
    {
        "passage": "Newton's second law relates force, mass, and acceleration.",
        "stem": "If you double the mass of a moving object while keeping the net force the same, what happens to its acceleration?",
        "options": ["It is halved", "It doubles", "It stays the same", "It becomes zero"],
        "topic": "physics",
    },
    {
        "passage": "Centripetal force keeps an object moving in a circular path.",
        "stem": "If you swing a stone on a string in a horizontal circle and the string breaks, in what direction will the stone fly off?",
        "options": ["Tangent to the circle (perpendicular to the radius)", "Toward the center", "Directly outward from the center", "Continue circling without the string"],
        "topic": "physics",
    },
    # Chemistry basics
    {
        "passage": "An acid is a substance that donates protons in solution.",
        "stem": "Acids generally turn blue litmus paper to which color?",
        "options": ["Red", "Green", "Yellow", "Black"],
        "topic": "chemistry",
    },
    {
        "passage": "Water is a chemical compound made of hydrogen and oxygen.",
        "stem": "What is the chemical formula of water?",
        "options": ["H2O", "HO2", "H2O2", "OH"],
        "topic": "chemistry",
    },
    {
        "passage": "The atmosphere of Earth is a mixture of gases.",
        "stem": "Which gas makes up the largest fraction of Earth's atmosphere?",
        "options": ["Nitrogen", "Oxygen", "Carbon dioxide", "Argon"],
        "topic": "chemistry",
    },
    # Biology basics
    {
        "passage": "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
        "stem": "Which gas is consumed by plants during photosynthesis?",
        "options": ["Carbon dioxide", "Oxygen", "Nitrogen", "Helium"],
        "topic": "biology",
    },
    {
        "passage": "Mammals are a class of vertebrates characterized by warm-blooded metabolism.",
        "stem": "Which of the following is NOT a mammal?",
        "options": ["Crocodile", "Whale", "Bat", "Dolphin"],
        "topic": "biology",
    },
    # Light / optics
    {
        "passage": "A convex mirror curves outward and is used in many vehicles.",
        "stem": "Convex mirrors are used as side-view mirrors on vehicles. The main advantage over a flat mirror is:",
        "options": ["Wider field of view, reducing blind spots", "Easier to clean", "Clearer image", "Cheaper to produce"],
        "topic": "optics",
    },
    {
        "passage": "A magnifying glass focuses sunlight to a single point at its focal distance.",
        "stem": "If the wood under a magnifying glass starts to smoke when the lens is held 12 inches above it, at what distance is the focal point?",
        "options": ["12 inches", "6 inches", "24 inches", "Cannot tell"],
        "topic": "optics",
    },
    # Pressure
    {
        "passage": "Air pressure inside a balloon depends on the amount of air inside it.",
        "stem": "Two identical balloons. Balloon A is fully inflated, balloon B is half inflated. Which has more air pressure inside?",
        "options": ["Balloon A", "Balloon B", "Equal pressure", "Neither has any pressure"],
        "topic": "physics",
    },
    {
        "passage": "Hot air rises because heated gas is less dense than cold gas.",
        "stem": "On a cold day, the front door of a heated house opens. Which way does the cold outside air flow into the house?",
        "options": ["Along the floor (cold air sinks)", "Along the ceiling", "Equally everywhere", "Upward through the chimney"],
        "topic": "physics",
    },
    # Bridges / architecture
    {
        "passage": "When a bridge is loaded, it deflects (sags) at certain points.",
        "stem": "Two identical cars are placed on a long simply-supported bridge of uniform thickness. Which part of the bridge experiences the greatest downward deflection?",
        "options": ["The middle of the bridge", "Below the cars", "Under the supports", "All parts deflect equally"],
        "topic": "structural",
    },
    # Electrical safety / common sense
    {
        "passage": "Friction can convert mechanical energy into other forms.",
        "stem": "When you rub your hands together in the cold, the friction generates:",
        "options": ["Heat", "Light", "Sound only", "Electricity"],
        "topic": "physics",
    },
    # Tools / mechanical advantage
    {
        "passage": "Bolt cutters use a long handle so that a small input force is amplified at the cutting jaws.",
        "stem": "Bolt cutters typically have very long handles. The main reason is:",
        "options": ["Mechanical advantage to cut thick bolts", "Aesthetic appeal", "Easier to store", "Stronger when dropped"],
        "topic": "mechanical",
    },
    # Constant-distance tracking
    {
        "passage": "Two vehicles travel in the same direction along a road.",
        "stem": "On a motorway, a car follows a fire engine moving at 60 mph. The distance between the two vehicles is constant. What is the car's speed?",
        "options": ["60 mph", "70 mph", "80 mph", "50 mph"],
        "topic": "mechanical",
    },
    # Temperature / states
    {
        "passage": "Water can exist in three states: solid, liquid, and gas.",
        "stem": "At standard atmospheric pressure, at what temperature does water boil?",
        "options": ["100 degrees Celsius", "0 degrees Celsius", "50 degrees Celsius", "212 degrees Celsius"],
        "topic": "physics",
    },
    # Geometry of solids
    {
        "passage": "The volume of a cylinder depends on its radius and height.",
        "stem": "Two granaries have the same height but different radii. The granary with the larger radius can:",
        "options": ["Hold more wheat", "Hold less wheat", "Hold the same amount of wheat", "Cannot be compared"],
        "topic": "geometry",
    },
    # Pendulum
    {
        "passage": "The period of a simple pendulum depends on its length, not on the mass of the bob.",
        "stem": "Two pendulums hang from a ceiling. Pendulum A has a shorter string than Pendulum B. Which pendulum will swing back and forth faster?",
        "options": ["Pendulum A", "Pendulum B", "Both swing at the same rate", "Cannot be determined"],
        "topic": "physics",
    },
    # Falling
    {
        "passage": "All objects in free fall experience the same gravitational acceleration, ignoring air resistance.",
        "stem": "Two identical balls are launched from the same height. Ball A is thrown straight outward (horizontal) at high speed; Ball B is dropped from rest. Ignoring air resistance, which lands first?",
        "options": ["Both land at the same time", "Ball A lands first", "Ball B lands first", "It depends on the ball mass"],
        "topic": "physics",
    },
    # Aerodynamics
    {
        "passage": "Air resistance on a moving object increases with the cross-section it presents to the airflow.",
        "stem": "Two birds are flying at the same speed. Bird A keeps its wings tucked back close to its body; Bird B holds its wings spread wide. Which bird experiences less air resistance?",
        "options": ["Bird A", "Bird B", "Both experience the same air resistance", "Cannot be determined"],
        "topic": "physics",
    },
]


# Sub-template family: L129-style careful converse / "Insufficient" reasoning
_CONVERSE_PASSAGES = [
    {
        "passage": "The flags of all Slavic countries display the colors red, blue, and white.",
        "stem": "The United Kingdom flag is red, blue, and white. Therefore, the United Kingdom is a Slavic country.",
        "answer": "Insufficient information",
        "distractors": ["True", "False"],
        "topic": "logic_converse",
        "extra_dist": ["Cannot be determined"],
    },
    {
        "passage": "Every cat in our garden has a collar.",
        "stem": "An animal in our garden has a collar. Therefore, it is a cat.",
        "answer": "Insufficient information",
        "distractors": ["True", "False"],
        "topic": "logic_converse",
        "extra_dist": ["Cannot be determined"],
    },
    {
        "passage": "Every athlete in the marathon wears a race number.",
        "stem": "Sam wears a race number. Therefore, Sam is an athlete in the marathon.",
        "answer": "Insufficient information",
        "distractors": ["True", "False"],
        "topic": "logic_converse",
        "extra_dist": ["Cannot be determined"],
    },
]


_INSTRUCTION_TAILS_TEXT = [
    "Reply with the letter (A/B/C/D) in <answer>...</answer>.",
    "Letter (A/B/C/D) in <answer>...</answer>.",
    "Place the single letter in <answer>X</answer>.",
]
_INSTRUCTION_TAILS_TF = [
    "Reply with the letter (A/B/C) in <answer>...</answer>.",
    "Letter (A/B/C) in <answer>...</answer>.",
    "Place the single letter (A/B/C) in <answer>X</answer>.",
]


class PassageFactualRecallQA(StandaloneVisualEnv):
    ENV_NAME = "passage_factual_recall"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0-L1: factual recall only (highest signal-to-noise).
        # L2-L4: mix factual + 1-clause converse trap.
        # L5-L9: mix factual + multi-clause converse trap.
        if level <= 1:
            return {"include_converse": False, "n_options": 4}
        if level <= 4:
            return {"include_converse": True, "n_options": 4, "converse_rate": 0.20}
        return {"include_converse": True, "n_options": 4, "converse_rate": 0.30}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 9941 + level * 313 + 7)

        use_converse = (
            cfg.get("include_converse", False) and
            rng.random() < cfg.get("converse_rate", 0.0)
        )

        if use_converse:
            return self._gen_converse(rng, cfg)
        return self._gen_factual(rng, cfg)

    def _gen_factual(self, rng, cfg) -> Optional[Tuple[str, str, Image.Image]]:
        item = rng.choice(_FACTUAL_BANK)
        passage = item["passage"]
        stem = item["stem"]
        # First option is correct in the bank; shuffle to randomize letter.
        opts_pool = list(item["options"])
        correct = opts_pool[0]
        distractors = opts_pool[1:]
        # Limit to n_options
        n_opts = cfg.get("n_options", 4)
        if len(distractors) > n_opts - 1:
            distractors = distractors[:n_opts - 1]
        all_opts = [correct] + distractors
        while len(all_opts) < n_opts:
            all_opts.append(f"None of the above")
        rng.shuffle(all_opts)
        ans_letter = chr(ord("A") + all_opts.index(correct))
        # Build option block
        opt_block = " ".join(f"({chr(ord('A') + i)}) {opt}"
                             for i, opt in enumerate(all_opts))
        tail = rng.choice(_INSTRUCTION_TAILS_TEXT)
        question = (
            f"Read the passage in the image and answer the multiple-choice "
            f"question.\n\n"
            f"{stem}\n\n"
            f"Options: {opt_block} {tail}"
        )
        img = self._render_passage(passage, rng)
        return question, ans_letter, img

    def _gen_converse(self, rng, cfg) -> Optional[Tuple[str, str, Image.Image]]:
        item = rng.choice(_CONVERSE_PASSAGES)
        passage = item["passage"]
        stem = item["stem"]
        # 3-option MCQ for converse: True / False / Insufficient
        all_opts = [item["answer"]] + list(item["distractors"])
        # Ensure 3 options
        if len(all_opts) > 3:
            all_opts = all_opts[:3]
        rng.shuffle(all_opts)
        ans_letter = chr(ord("A") + all_opts.index(item["answer"]))
        opt_block = " ".join(f"({chr(ord('A') + i)}) {opt}"
                             for i, opt in enumerate(all_opts))
        tail = rng.choice(_INSTRUCTION_TAILS_TF)
        question = (
            f"Read the rule in the image and decide whether the following "
            f"statement logically follows.\n\n"
            f"Statement: {stem}\n\n"
            f"Options: {opt_block} {tail}"
        )
        img = self._render_passage(passage, rng)
        return question, ans_letter, img

    # ---------------------------------------------------------------- #

    def _render_passage(self, passage: str, rng: random.Random) -> Image.Image:
        """Render a short passage as text-in-image (reference 'common sense ocr' style)."""
        # Wrap to ~50 chars/line
        words = passage.split()
        lines = []
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 > 55 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            lines.append(cur)

        n_lines = len(lines)
        max_line_len = max((len(l) for l in lines), default=10)
        fig_w = max(5.0, min(11, 0.085 * max_line_len + 1.0))
        fig_h = max(2.0, 0.55 * n_lines + 1.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        title_y = 0.95
        ax.text(0.5, title_y, "Passage", fontsize=14, ha="center", va="top",
                fontweight="bold", color="#212121")

        line_y = 0.84
        line_step = 0.74 / max(n_lines, 1)
        for i, line in enumerate(lines):
            ax.text(0.05, line_y - i * line_step, line, fontsize=12,
                    ha="left", va="top", color="#212121",
                    family="DejaVu Sans")
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_factual"
    os.makedirs(out_dir, exist_ok=True)
    env = PassageFactualRecallQA()
    for level in (0, 3, 6, 9):
        for seed in range(5):
            ok = env.generate(seed=seed * 17 + 1, parameter={"level": level})
            if not ok:
                print(f"L{level} s{seed}: FAILED")
                continue
            env.render().save(os.path.join(out_dir, f"L{level}_s{seed}.png"))
            v = env.verify(f"<answer>{env._answer}</answer>")
            v2 = env.verify(env._answer)
            v3 = env.verify("<answer>Z</answer>")
            print(f"L{level} s{seed}: ans={env._answer} verify={v['accuracy']} bare={v2['accuracy']} wrong={v3['accuracy']}")
