"""Flowchart QA — diversity + difficulty redesign 2026-04-16.

Round-2 fix: Grade D difficulty + low diversity.
- Box shape pool expanded: rect, rounded-rect, diamond, oval, hexagon, parallelogram.
- Per-seed palette shuffle, background colour variance.
- 4+ question-phrasing variants per operation.
- Layout variance: vertical (top-down) or branching L/R per-seed.
- Difficulty gradient uses structurally different operations per level.

Levels:
  L0-L1: trace_variable (linear chain of 2-3 ops) / which_branch (single decision).
  L2-L3: two_branch_output (1 decision + 2 ops per branch).
  L4-L5: three_decision_output (3 nested decisions).
  L6-L7: reverse_engineer_input / loop_iteration_count.
  L8   : multi_variable_trace (x, y, z with 3-4 ops).
  L9   : conditional_accumulator / nested_loop_count (compound hardest).
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

_BOX_COLOURS = [
    "#3498db", "#5dade2", "#27ae60", "#58d68d", "#e67e22", "#d35400",
    "#8e44ad", "#af7ac5", "#16a085", "#f1c40f", "#2980b9",
]
_DECISION_COLOURS = ["#f39c12", "#e67e22", "#d6a40e", "#e74c3c", "#c47c26"]
_TERMINAL_COLOURS = ["#27ae60", "#e74c3c", "#2ecc71", "#16a085"]

_TITLE_POOL = [
    "Follow the flowchart", "Flowchart Trace", "Program Flow",
    "Process Diagram", "Execution Path", "Control Flow",
]

_Q_TRACE = [
    "Starting with x = {x}, trace through the flowchart. What is the final value of x?",
    "Given initial x = {x}, follow the operations in order. Output?",
    "With x = {x} as input, apply each step. What x is output?",
    "Follow the flowchart with x = {x}. Report the final value printed.",
    "Trace the flowchart from x = {x} and report the final value.",
    "Input x = {x} - perform each operation; what integer is output?",
    "With starting value x = {x}, execute all steps. Final x?",
    "Apply each box's operation starting from x = {x}. Output the final x.",
    "Given x = {x} at the START oval, what integer is printed at the end?",
    "Begin at x = {x}; step through every operation box. Return the final x.",
    "For x = {x}, compute the x that emerges at the OUTPUT box.",
    "Start the program with x = {x}. What value does 'Output x' print?",
    "Trace through the flowchart with starting x = {x}. Integer answer only.",
    "What is the result of running the flowchart with input x = {x}?",
    "Simulate the flowchart for input x = {x}; report the printed integer.",
    "x begins at {x}; after all operations, what is x?",
]

_Q_WHICH_BRANCH = [
    "Given input x = {val}, which path (A or B) does the flowchart take?",
    "With x = {val} entering the flowchart, does it follow path A or B?",
    "For x = {val}, which branch does the flowchart output - A or B?",
    "Trace the flowchart with x = {val}. Report the final path label (A or B).",
    "Input x = {val}: which terminal ('A' or 'B') is reached?",
    "Starting at x = {val}, does the flowchart output 'A' or 'B'?",
    "For input x = {val}, which of the two paths (A / B) is taken?",
    "Given x = {val}, decide whether path A or B is executed.",
    "With x = {val}, which of the two endpoints is printed: A or B?",
    "Which terminal label (A or B) does the flowchart reach for x = {val}?",
    "x = {val} flows into the diamond; which branch output is produced - A or B?",
    "Decide A vs B based on the decision diamond for input x = {val}.",
    "Follow the flowchart with x = {val} and return the reached label (A or B).",
    "For x = {val}, what path - A or B - does the flowchart take?",
    "When x = {val} is input, which of 'A' / 'B' is the final output?",
    "Determine the output label (A or B) when x = {val}.",
]

_Q_TWO_BRANCH = [
    "Starting with x = {val}, trace the flowchart; what value is output?",
    "With x = {val}, follow the decision then the operations; report the result.",
    "Input x = {val} into the flowchart; the branching is as shown. Return the output value.",
    "Follow the flow with x = {val} and give the final integer output.",
    "Trace the flow for x = {val}; after the branch executes, what is the output?",
    "For input x = {val}, evaluate the shown decision and ops; return the integer.",
    "x = {val} enters the diamond, then one of two operation chains runs. Output?",
    "Given x = {val}, apply the correct branch's ops and report the output.",
    "With x = {val}, the flowchart picks a branch based on the condition; final value?",
    "Run the flowchart for x = {val}; integer output?",
    "Starting x = {val}, simulate the two-branch flow. Return the printed integer.",
    "For x = {val}, follow the diamond to the correct branch and return the result.",
    "Simulate the flowchart from x = {val}; what number is output at the end?",
    "Given input x = {val}, the diamond routes to one branch; compute the final x.",
    "Input value x = {val}: trace the appropriate branch; return final x.",
    "With x = {val}, determine the branch taken and compute the output integer.",
]

_Q_THREE_DECISION = [
    "With input x = {x}, trace through the three-decision flowchart. What value is output?",
    "Starting x = {x}; apply the nested conditions shown. Return the output integer.",
    "Follow the flow from x = {x} through each decision; report the final printed value.",
    "Given x = {x}, follow the three-decision cascade; what is output?",
    "Input x = {x}: trace the three-level decision tree; output integer?",
    "For x = {x}, follow each diamond in turn and report the output.",
    "With x = {x} entering the nested decisions, what is the final output?",
    "Trace x = {x} through three nested conditionals. Return the output.",
    "x = {x} flows through three decision diamonds; what integer is printed?",
    "Given initial x = {x}, evaluate the three-way branching and print the result.",
    "Simulate the three-decision flow with x = {x}; return the output value.",
    "From x = {x}, apply each diamond's condition; output the final integer.",
    "Input x = {x}: what value comes out of the three-branch flowchart?",
    "Given x = {x} at start, what does the three-decision chart output?",
    "For starting x = {x}, trace all three decisions and give the final x.",
    "With x = {x}, which branch wins at each diamond? Return the final output value.",
]

_Q_REVERSE = [
    "The flowchart outputs {target}. What was the starting value of x?",
    "Given the flowchart produces {target} as output, determine the input x (integer).",
    "To get output {target} from this flowchart, what integer input x is required?",
    "Reverse-engineer the input x that yields output {target}.",
    "If the flowchart's output is {target}, what was x?",
    "Work backwards: output = {target}. Find input x.",
    "The output is {target} - what integer did x start at?",
    "Given output {target}, deduce the starting x.",
    "Find the integer input x such that the flowchart outputs {target}.",
    "Starting x = ? produces {target}. What value of x gives this result?",
    "For output {target}, invert the operations to recover x.",
    "What x makes this flowchart print {target}?",
    "Determine the pre-image x given output {target}.",
    "The printed value is {target}; recover the original input x.",
    "If x must be chosen so the output is {target}, what integer x works?",
    "Given output = {target}, backsolve the operations for input x.",
]

_Q_LOOP_VISIBLE = [
    "Starting with x = {start}, the loop adds {inc} each iteration while x < {limit}. How many iterations run?",
    "How many times does the loop body execute? x starts at {start}, adds {inc}, stops when x >= {limit}.",
    "The loop iterates while x < {limit}, starting from x = {start} and adding {inc}. Iteration count?",
    "Count the iterations of the loop: x = {start}, x += {inc}, while x < {limit}.",
    "With x = {start}, incrementing by {inc}, running until x >= {limit}: how many iterations?",
    "x begins at {start} and grows by {inc} per step until x >= {limit}. Total iterations?",
    "Given start {start}, step {inc}, limit {limit}: how many loop passes execute?",
    "Loop: x = {start}; while x < {limit}: x = x + {inc}. Count the iterations.",
    "How many loop iterations with x = {start}, step {inc}, stop-condition x >= {limit}?",
    "For start x = {start}, add {inc} each pass, stop when x >= {limit} - iteration count?",
    "How many times does x = x + {inc} execute, given x starts at {start} and stops at x >= {limit}?",
    "Given start={start}, step={inc}, limit={limit}: how many loop body executions?",
    "The loop body runs while x < {limit} (start {start}, step {inc}). Body run count?",
    "x = {start}; each iteration x += {inc}; loop ends when x >= {limit}. Iteration tally?",
    "Tally the iterations of this loop: start {start}, increment {inc}, upper cutoff {limit}.",
    "Count loop iterations: x0 = {start}, step = {inc}, stop condition x >= {limit}.",
]

_Q_LOOP_HIDDEN = [
    "The loop increments x by a fixed step (visible in the flowchart) while x is below the threshold in the decision diamond. Read the start, step, and threshold from the image and count how many iterations execute.",
    "Read the start, step, and limit from the flowchart. How many iterations of the loop body execute?",
    "Count the iterations of the loop based only on the start, step, and limit values shown in the flowchart.",
    "Inspect the flowchart for the start value, increment, and exit condition; count total loop iterations.",
    "The flowchart's START oval shows the initial x. The diamond shows the limit; the body shows the step. Iteration count?",
    "How many iterations of the increment-while-less-than loop run? All numeric values are only on the image.",
    "Using only values shown in the flowchart, count how many times the loop body executes.",
    "Read start, step, and bound from the image. How many iterations run before exiting?",
    "Determine the number of loop iterations - initial x, increment, and limit are all visible in the flowchart.",
    "Counter loop iterations: start/step/limit are in the image. Give the iteration count.",
    "The loop uses only on-image values for start, step, and bound. Count the iterations.",
    "Read all three numeric parameters (start, step, limit) from the flowchart, then report iteration count.",
    "Check the START oval, the body box, and the decision diamond for the three numbers; count iterations.",
    "The flowchart fully specifies start, step, and exit threshold. Count the number of iterations.",
    "How many loop body runs happen? All necessary values must be read from the image.",
    "Take the start value, step, and exit bound from the flowchart; return iteration count.",
]

_Q_MULTI_VISIBLE = [
    "With x = {x} and y = {y}, follow the ops. What is final x + y?",
    "Initially x = {x}, y = {y}; after the operations, compute x + y.",
    "Starting x = {x}, y = {y}; return the sum x + y after all ops.",
    "Given x = {x}, y = {y} initially, trace the flow. Output x + y.",
    "With x = {x}, y = {y}, run each step and return the final x + y.",
    "x begins at {x}, y at {y}; after all operations, compute x + y.",
    "Starting (x, y) = ({x}, {y}), execute the operations. Return x + y.",
    "x = {x}, y = {y}: follow the flow and output x + y at the end.",
    "Initial values x = {x}, y = {y}. What is x + y after all ops?",
    "Starting state: x = {x}, y = {y}. Compute x + y after the program runs.",
    "For (x, y) = ({x}, {y}), trace the flow; final x + y = ?",
    "With x = {x} and y = {y} at start, return x + y after execution.",
    "Trace the program from x = {x}, y = {y}; report final x + y.",
    "Execute the ops with x = {x}, y = {y}; output x + y.",
    "Given initial x = {x}, y = {y}, compute x + y after the operations complete.",
    "Start x = {x}, y = {y}; after the chain of ops, what is x + y?",
]

_Q_MULTI_HIDDEN = [
    "Read the initial x and y values from the flowchart's START oval, then trace the operations. What is the final x + y?",
    "The flowchart's START oval shows the initial x and y values. Trace through the operations and output final x + y.",
    "Initial x and y are in the flowchart's first oval - trace the ops and return x + y.",
    "The starting values of x and y are shown only in the flowchart. Trace the operations and return x + y.",
    "From the START oval's (x, y), run the operations and output x + y.",
    "Use the initial (x, y) inside the START oval. Trace each op; return final x + y.",
    "Initial x and y can be read from the flowchart. After the ops run, what is x + y?",
    "The flowchart specifies start (x, y). Execute the operations and report x + y.",
    "Find the initial x, y values in the first oval; trace the flow; output x + y.",
    "Locate initial x and y in the flowchart, trace through every op, return x + y.",
    "The two starting variables (x, y) appear only in the flowchart. Report x + y after ops.",
    "Extract start values x and y from the START oval; simulate; print x + y.",
    "Start values of x and y live inside the image. Trace the ops and output x + y.",
    "Only the image shows x and y's starting values. After all ops, what is x + y?",
    "Starting (x, y) come from the first oval. Trace each operation; output x + y.",
    "Read x and y from the flowchart's starting oval; trace; return the final x + y.",
]

_Q_DEEP_VISIBLE = [
    "Starting x = {x}, y = {y}, z = {z}, trace through ops. Return final x * y.",
    "Given x = {x}, y = {y}, z = {z}, after the ops the output is x * y. Compute.",
    "x = {x}, y = {y}, z = {z} initially; follow the operations and output x * y.",
    "Initial (x, y, z) = ({x}, {y}, {z}). After all ops, return x * y.",
    "With x = {x}, y = {y}, z = {z}, evaluate every operation and output x * y.",
    "Starting values x = {x}, y = {y}, z = {z}; compute x * y after running the flow.",
    "From (x, y, z) = ({x}, {y}, {z}), execute all ops and return x * y.",
    "Initial x = {x}, y = {y}, z = {z}; trace the program; output x * y.",
    "Given x = {x}, y = {y}, z = {z} at start, what is x * y at the end?",
    "Execute the flowchart starting from x = {x}, y = {y}, z = {z}; report x * y.",
    "Run the program: x = {x}, y = {y}, z = {z}. Return final x * y.",
    "With initial state x = {x}, y = {y}, z = {z}, output x * y after all operations.",
    "Trace from (x, y, z) = ({x}, {y}, {z}); what is the product x * y after the flow?",
    "Start at x = {x}, y = {y}, z = {z}; follow the ops; compute x * y.",
    "Given starting x = {x}, y = {y}, z = {z}, simulate and output x * y.",
    "For initial x = {x}, y = {y}, z = {z}, what is the final x * y?",
]

_Q_DEEP_HIDDEN = [
    "The flowchart's first oval contains the starting x, y, z values. Trace through all operations (including any conditional branch) and output the final x * y.",
    "Read the initial values of x, y, z from the START oval. Execute the ops (noting the decision diamond branching). Return final x * y.",
    "Starting values for x, y, z are only shown inside the flowchart. Trace the ops - including evaluating any if-condition - and report final x * y.",
    "Initial x, y, z appear only in the START oval. Trace each op (including any conditional) and output x * y.",
    "Read (x, y, z) from the first oval, run each operation including conditionals, return x * y.",
    "The starting triple (x, y, z) lives in the START oval. Execute everything and return x * y.",
    "Use the flowchart's first oval for initial x, y, z; trace through; output x * y.",
    "Extract initial x, y, z from the image; evaluate every step including the if-branch; return x * y.",
    "Initial x, y, z are inside the flowchart - trace all ops and conditionals; output x * y.",
    "The flowchart's START oval is the only source of initial values; trace; return x * y.",
    "Find starting x, y, z in the flowchart, run the program step-by-step, output x * y.",
    "Start values x, y, z come from the first oval. Execute each op and any conditional; return x * y.",
    "Read x, y, z from the START oval only. Trace the flow and compute final x * y.",
    "All initial values live inside the image. Trace ops and conditionals; output x * y.",
    "The first oval provides x, y, z; follow the operations; return the final x * y.",
    "Using the START oval's x, y, z, trace through the flowchart; report x * y.",
]

_Q_NESTED_VISIBLE = [
    "The flowchart accumulates sum = 1 + 2 + ... + i while i <= {n}. What is sum?",
    "Compute sum = 1 + 2 + ... + {n} via the shown loop. Return sum.",
    "The loop sums integers 1 through {n} into the variable sum. Final sum?",
    "What is the value of sum after the loop finishes (i goes from 1 to {n})?",
    "Loop runs for i = 1 to {n}; sum += i each iteration. Final sum?",
    "The sum 1 + 2 + ... + {n} is accumulated; output the final sum.",
    "Sum integers from 1 through {n} per the loop. Return final sum.",
    "For i = 1..{n}, accumulate into sum. What is sum after the loop?",
    "The loop increments i from 1 to {n}, adding each to sum. Output sum.",
    "Trace the loop: sum adds i while i <= {n}. Final sum value?",
    "Given the loop iterates i from 1 to {n}, return the accumulated sum.",
    "Integers 1..{n} are summed by the loop. Report sum.",
    "What integer value does 'sum' hold after the loop terminates? (i goes 1 to {n})",
    "The flowchart's loop adds i from 1 to {n} into sum. Output the final sum.",
    "Compute 1 + 2 + ... + {n}, as the shown loop does. Answer?",
    "With the loop running for i in 1..{n} and summing all values, what is the final sum?",
]

_Q_NESTED_HIDDEN = [
    "Read the loop condition bound and the starting value from the flowchart. Compute sum = 1 + 2 + ... + N for N taken from the image. Return sum.",
    "The flowchart accumulates sum = 1 + 2 + ... + i while i <= N, with N visible only in the image. Return final sum.",
    "Determine N from the loop's decision diamond, then return 1 + 2 + ... + N as the final sum.",
    "Read the upper bound N from the flowchart's diamond, then compute 1 + 2 + ... + N.",
    "The flowchart sums i from 1 to N where N is on the image; return the final sum.",
    "Read N from the decision diamond and output 1 + 2 + ... + N.",
    "Use the flowchart's bound N to compute sum = 1 + 2 + ... + N.",
    "The loop sums integers up to N (find N in the image). Return the final sum.",
    "Inspect the diamond for N, then compute 1 + 2 + ... + N.",
    "Determine the loop's terminal N from the diamond; output 1 + 2 + ... + N.",
    "Figure out N from the flowchart, then add integers 1..N into sum. Final sum?",
    "Read the upper limit N from the flowchart; return sum = 1 + 2 + ... + N.",
    "N is shown only in the diamond; report the sum 1 + 2 + ... + N.",
    "Extract N from the decision diamond. Return 1 + 2 + ... + N.",
    "The sum 1 + 2 + ... + N is accumulated, with N in the image. Report the final sum.",
    "Read the bound N (only in the flowchart) and compute 1 + 2 + ... + N.",
]

_Q_COND_VISIBLE = [
    "x starts at {start}, increments by {step} while x <= {limit}. It adds x to sum only when x is odd. Final sum?",
    "Starting x = {start}, step {step}, limit {limit}; add x to sum iff x is odd. Return sum.",
    "Trace the loop: x = {start}, increments {step}, while x <= {limit}; sum accumulates odd x's. Output sum?",
    "x begins at {start}. Each iteration: if x is odd, sum += x; then x += {step}; repeat while x <= {limit}. Sum?",
    "With start={start}, step={step}, limit={limit}, odd-only accumulator. Final sum?",
    "The loop runs x from {start} while x <= {limit}, step {step}, summing only odd x. Output sum.",
    "Trace an odd-only accumulator: start {start}, step {step}, bound {limit}. Final sum?",
    "x = {start}; step {step}; cap {limit}; add odd x's to sum. Return sum.",
    "For x starting {start}, incrementing {step}, running while x <= {limit}: accumulate odd x's. Sum?",
    "Start x at {start}, add {step} each loop, stop once x > {limit}, sum only odd values. Sum total?",
    "Compute the odd-value sum: x in [{start}, {limit}] stepping {step}. Return sum.",
    "Sum the odd x values as x iterates from {start} to <= {limit} by {step}. Final sum?",
    "Odd-only accumulator: x0 = {start}, step = {step}, bound {limit}. Return sum.",
    "Given start {start}, step {step}, limit {limit}, and odd-only condition, compute final sum.",
    "The flowchart adds x to sum only when x is odd; x goes from {start} to <= {limit} by {step}. Sum?",
    "Loop: x = {start}; while x <= {limit}: if x odd, sum += x; x += {step}. Return sum.",
]

_Q_COND_HIDDEN = [
    "Read the start x, step, and limit from the flowchart. The loop runs while x <= limit, accumulating x into sum only when x is odd. Report final sum.",
    "Trace the flowchart: it conditionally adds x to sum only when x is odd, then advances x by the labeled step until x exceeds the limit. What is sum?",
    "The starting x, increment, and loop bound all appear ONLY in the flowchart - use them to compute final sum (odd-only accumulator).",
    "Read the three numeric parameters (start, step, limit) from the flowchart and compute the odd-only accumulator sum.",
    "Extract start/step/limit from the flowchart; sum only odd x values as the loop runs.",
    "All three parameters (start, step, limit) are on the image. Run the odd-only accumulator and return sum.",
    "Find start, step, and limit in the flowchart; compute sum over odd x in the loop range.",
    "Use the image-visible start/step/limit to compute the odd-only accumulated sum.",
    "The odd-only accumulator's parameters are only shown in the flowchart; return final sum.",
    "Start, increment, and cap come from the flowchart. Sum only odd x's. Output sum.",
    "Read start x, step, and limit from the image. Accumulate odd x's; return sum.",
    "Simulate the odd-only accumulator using start/step/limit visible in the flowchart.",
    "The flowchart specifies all loop parameters; return sum of odd x values across the iterations.",
    "Inspect the flowchart for start x, step size, and upper limit; compute the odd-only sum.",
    "Parse start, step, and bound from the image and compute the odd-only sum.",
    "Using only flowchart-visible values for start/step/limit, return the odd-only accumulator's final sum.",
]

_Q_DEEP_PROG = [
    "Starting x, y, z, i and sum values are visible in the flowchart's START oval. Execute the program faithfully until either i reaches the loop bound OR z hits 0 (the break). Then report the final value of 'sum + x - y' as computed in the OUTPUT box. Integer answer.",
    "Trace the flowchart very carefully: the loop body has FIVE steps (conditional update of x or y, decrement z, conditional accumulate to sum, break-if-z-zero, increment i). Starting values are in the START oval. What integer value is output?",
    "Simulate the program until the loop terminates. The break condition (z=0) may stop execution before the counter bound. Return the final output expression 'sum + x - y' as an integer.",
    "Read x, y, z, i, sum from the START oval. Simulate the 5-step loop until z=0 or i=N. Return sum + x - y.",
    "Execute the flowchart's outer loop (body: conditional x/y update, decrement z, conditional sum+=x/y, break-if-z=0, i+=1). Return sum + x - y after termination.",
    "Carefully simulate: the 5-step loop body has two diamonds and a break. Starting vars are on the image. Output sum + x - y.",
    "Run the program to completion respecting both the i-bound and the z=0 break. Return sum + x - y.",
    "Trace the long flowchart - 5 ops per iteration, break on z=0. Return sum + x - y.",
    "Simulate the loop with break-on-zero and conditional x-vs-y update. Compute sum + x - y.",
    "Run the 5-step inner loop until either counter expires or z hits 0. Report sum + x - y.",
    "The program's state is entirely on the image. Execute faithfully and return sum + x - y.",
    "Execute each of the 5 loop-body steps per iteration until break or counter. Output sum + x - y.",
    "Simulate to termination (i>=N OR z=0). Return the integer sum + x - y.",
    "Faithfully trace every iteration; two diamonds and a break govern each pass. Return sum + x - y.",
    "Step through the program reading START oval values from the image. Return sum + x - y after the loop ends.",
    "Run the flowchart to termination (break on z=0 may fire early) and return the integer sum + x - y.",
]

class FlowchartQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "flowchart"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # Iter-3: L9 now uses brand-new `deep_program_trace` handler with
        # 9+ operations including goto/loops and mutual x/y/z interactions.
        base = {
            0: {"qtypes": ["trace_variable", "which_branch"]},
            1: {"qtypes": ["trace_variable", "which_branch"]},
            2: {"qtypes": ["two_branch_output"]},
            3: {"qtypes": ["two_branch_output", "trace_variable"]},
            4: {"qtypes": ["three_decision_output"]},
            5: {"qtypes": ["three_decision_output", "reverse_engineer_input"]},
            6: {"qtypes": ["reverse_engineer_input", "loop_iteration_count"]},
            7: {"qtypes": ["loop_iteration_count", "multi_variable_trace"]},
            8: {"qtypes": ["multi_variable_trace", "deep_multi_variable_trace"]},
            9: {"qtypes": ["deep_program_trace"]},
        }[level]
        # At L6-L9, hide the numeric input inside the flowchart's START box
        # (the question states the input, so the model must trace through
        # the operation boxes carefully without a visible starting value).
        # We also add decoy/dead-end branches at L8-L9.
        base["hide_input_in_box"] = level >= 6
        base["add_decoy_branches"] = level >= 8
        return base

    def _generate_problem(self, seed, parameter):
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 2039)
        style_rng = random.Random((self.seed or 0) * 1000 + level * 53 + 733)
        qtype = rng.choice(cfg["qtypes"])
        self._flow_cfg = cfg  # tuck cfg on instance for handlers

        dispatch = {
            "trace_variable": self._trace_variable,
            "which_branch": self._which_branch,
            "two_branch_output": self._two_branch_output,
            "three_decision_output": self._three_decision_output,
            "reverse_engineer_input": self._reverse_engineer_input,
            "loop_iteration_count": self._loop_iteration_count,
            "multi_variable_trace": self._multi_variable_trace,
            "deep_multi_variable_trace": self._deep_multi_variable_trace,
            "nested_loop_count": self._nested_loop_count,
            "conditional_accumulator": self._conditional_accumulator,
            "deep_program_trace": self._deep_program_trace,
        }
        fn = dispatch.get(qtype)
        if fn is None:
            return None
        for _ in range(10):
            try:
                r = fn(rng, style_rng)
                if r is not None:
                    return r
            except Exception:
                continue
        return None

    def _maybe_hide_input(self, label):
        """Deprecated after 2026-04-17: we now keep values on the image
        (they are the SOURCE OF TRUTH) and remove them from the question
        text instead. This method is a no-op for compatibility."""
        return label

    def _add_decoy_branches(self, ax, style_kit, rng, bounds):
        """Draw 1-2 small decoy boxes connected by dashed 'dead-end' arrows.
        Used at L8-L9 to add visual clutter. bounds = (x_min, x_max, y_min, y_max)
        should be a region known NOT to overlap main flow content; callers
        must pass safe bounds (e.g., far-right margin)."""
        if not getattr(self, "_flow_cfg", {}).get("add_decoy_branches"):
            return
        x_min, x_max, y_min, y_max = bounds
        if x_max - x_min < 0.8 or y_max - y_min < 0.6:
            return  # not enough room — skip to avoid overlap
        for _ in range(rng.randint(1, 2)):
            dx = rng.uniform(x_min + 0.2, x_max - 0.3)
            dy = rng.uniform(y_min + 0.2, y_max - 0.3)
            decoy_text = rng.choice(["DEAD END", "skip", "NOP",
                                     "deprecated", "(ignore)"])
            self._draw_box(ax, dx, dy, decoy_text, style_kit, shape="oval",
                           w=1.0, h=0.4)
            # Dashed arrow to nowhere
            ax.plot([dx + rng.uniform(-0.15, 0.15),
                     dx + rng.uniform(-0.2, 0.2)],
                    [dy - 0.22, dy - 0.45 + rng.uniform(-0.05, 0.05)],
                    linestyle="--", color="#999999", linewidth=1.0, alpha=0.6)

    # ------------------------------------------------------------------ #
    # Drawing helpers with diversity
    # ------------------------------------------------------------------ #

    def _style_kit(self, style_rng):
        palette = list(_BOX_COLOURS)
        style_rng.shuffle(palette)
        return {
            "bg": style_rng.choice([
                "#ffffff", "#fafafa", "#f3f6fa", "#f7f9fb", "#fff8e7",
                "#fdfdfd", "#fbf8f2", "#f7f7f4", "#faf6ef", "#f4f4f0",
            ]),
            "box_colour": palette[0],
            "box_alt": palette[1],
            "diamond": style_rng.choice(_DECISION_COLOURS),
            "terminal": style_rng.choice(_TERMINAL_COLOURS),
            "arrow": style_rng.choice(["#4a4a4a", "#5b5b5b", "#2c3e50", "#34495e",
                                        "#1f2a36", "#222222", "#1a1a1a"]),
            "shape": style_rng.choice(["rect", "rounded_rect", "parallelogram"]),
            "title": style_rng.choice(_TITLE_POOL),
            "dpi": style_rng.choice([100, 105, 110, 115, 120]),
            "figsize_scale": style_rng.uniform(0.9, 1.15),
            "fontsize_scale": style_rng.uniform(0.92, 1.08),
            "edge_color": style_rng.choice(["#2c3e50", "#1f2a36", "#222222",
                                             "#1a1a1a", "#34495e"]),
        }

    def _draw_box(self, ax, x, y, text, style_kit, shape="process", w=2.0, h=0.7):
        ec = style_kit.get("edge_color", "#2c3e50")
        fs_scale = style_kit.get("fontsize_scale", 1.0)
        if shape == "oval":
            e = mpatches.Ellipse((x, y), w, h,
                                  facecolor=style_kit["terminal"],
                                  edgecolor=ec, linewidth=1.5, alpha=0.92)
            ax.add_patch(e)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=10 * fs_scale,
                    fontweight="bold", color="white")
        elif shape == "diamond":
            d = plt.Polygon(
                [(x, y + h/1.25), (x + w/1.6, y), (x, y - h/1.25), (x - w/1.6, y)],
                facecolor=style_kit["diamond"], edgecolor=ec,
                linewidth=1.5, alpha=0.92)
            ax.add_patch(d)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=9 * fs_scale,
                    fontweight="bold", color="black")
        elif shape == "hexagon":
            pts = [(x + w/1.4 * math.cos(math.radians(60 * i)),
                     y + h/1.1 * math.sin(math.radians(60 * i))) for i in range(6)]
            hx = plt.Polygon(pts, facecolor=style_kit["box_alt"],
                              edgecolor=ec, linewidth=1.5, alpha=0.92)
            ax.add_patch(hx)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=9 * fs_scale,
                    fontweight="bold", color="white")
        elif shape == "parallelogram":
            skew = h * 0.3
            pts = [(x - w/2 + skew, y - h/2), (x + w/2, y - h/2),
                    (x + w/2 - skew, y + h/2), (x - w/2, y + h/2)]
            p = plt.Polygon(pts, facecolor=style_kit["box_colour"],
                              edgecolor=ec, linewidth=1.5, alpha=0.92)
            ax.add_patch(p)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=9 * fs_scale,
                    fontweight="bold", color="white")
        else:  # rect or rounded_rect
            pad_style = "round,pad=0.1" if style_kit["shape"] == "rounded_rect" else "square,pad=0.1"
            r = mpatches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle=pad_style, facecolor=style_kit["box_colour"],
                edgecolor=ec, linewidth=1.5, alpha=0.92)
            ax.add_patch(r)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=9 * fs_scale,
                    fontweight="bold", color="white")

    def _draw_arrow(self, ax, x1, y1, x2, y2, style_kit, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=style_kit["arrow"], lw=1.6))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=9, color="#b71c1c",
                    fontweight="bold")

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    def _trace_variable(self, rng, style_rng):
        x = rng.randint(1, 12)
        ops_pool = ["add", "multiply", "subtract"]
        n_ops = rng.randint(2, 3)
        ops = [rng.choice(ops_pool) for _ in range(n_ops)]
        op_vals = [rng.randint(2, 8) for _ in ops]
        result = x
        op_texts = []
        for op, val in zip(ops, op_vals):
            if op == "add":
                result += val; op_texts.append(f"x = x + {val}")
            elif op == "multiply":
                result *= val; op_texts.append(f"x = x * {val}")
            else:
                result -= val; op_texts.append(f"x = x - {val}")

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(4.5 * _fs, (2 + n_ops * 1.3) * _fs))
        fig.patch.set_facecolor(sk["bg"])
        y = 2.5 + n_ops * 1.2
        self._draw_box(ax, 2, y, f"x = {x}", sk, shape="oval", w=2.0, h=0.8)
        prev_y = y
        for t in op_texts:
            y -= 1.2
            self._draw_box(ax, 2, y, t, sk, w=2.2, h=0.75)
            self._draw_arrow(ax, 2, prev_y - 0.4, 2, y + 0.4, sk)
            prev_y = y
        y -= 1.2
        self._draw_box(ax, 2, y, "Output x", sk, shape="oval", w=1.8, h=0.75)
        self._draw_arrow(ax, 2, prev_y - 0.4, 2, y + 0.4, sk)
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(y - 0.8, 3 + n_ops * 1.2)
        ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)

        sidx = (self.seed or 0) % 16
        q = _Q_TRACE[sidx].format(x=x)
        return q, str(result), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _which_branch(self, rng, style_rng):
        val = rng.randint(1, 30)
        threshold = rng.randint(8, 22)
        goes_yes = val > threshold

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(5.5 * _fs, 4.5 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 2.5, 3.8, f"Input: {val}", sk, shape="oval", w=1.8, h=0.7)
        self._draw_arrow(ax, 2.5, 3.45, 2.5, 2.85, sk)
        self._draw_box(ax, 2.5, 2.5, f"x > {threshold}?", sk, shape="diamond", w=2.2, h=0.9)
        # Yes right
        self._draw_arrow(ax, 3.9, 2.5, 4.6, 2.5, sk, "Yes")
        self._draw_box(ax, 4.6, 1.2, "Path A", sk, shape="oval", w=1.5, h=0.75)
        self._draw_arrow(ax, 4.6, 2.5 - 0.45, 4.6, 1.55, sk)
        # No left
        self._draw_arrow(ax, 1.1, 2.5, 0.4, 2.5, sk, "No")
        self._draw_box(ax, 0.4, 1.2, "Path B", sk, shape="oval", w=1.5, h=0.75)
        self._draw_arrow(ax, 0.4, 2.5 - 0.45, 0.4, 1.55, sk)
        ax.set_xlim(-0.6, 6); ax.set_ylim(0.3, 4.4); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        ans = "A" if goes_yes else "B"
        sidx = (self.seed or 0) % 16
        q = _Q_WHICH_BRANCH[sidx].format(val=val)
        return q, ans, self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _two_branch_output(self, rng, style_rng):
        val = rng.randint(1, 25)
        threshold = rng.randint(6, 18)
        # Yes: add then multiply; No: subtract then add
        if val > threshold:
            v_add = rng.randint(3, 10); v_mul = rng.randint(2, 4)
            result = (val + v_add) * v_mul
        else:
            v_sub = rng.randint(2, 8); v_add2 = rng.randint(3, 10)
            result = (val - v_sub) + v_add2
        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(6.5 * _fs, 6 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 3.3, 5.2, f"x = {val}", sk, shape="oval", w=1.8, h=0.7)
        self._draw_arrow(ax, 3.3, 4.85, 3.3, 4.25, sk)
        self._draw_box(ax, 3.3, 3.9, f"x > {threshold}?", sk, shape="diamond", w=2.2, h=0.9)
        # Yes
        self._draw_arrow(ax, 4.7, 3.9, 5.6, 3.9, sk, "Yes")
        if val > threshold:
            self._draw_box(ax, 5.6, 2.8, f"x = x + {v_add}", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 5.6, 3.5, 5.6, 3.15, sk)
            self._draw_box(ax, 5.6, 1.7, f"x = x * {v_mul}", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 5.6, 2.4, 5.6, 2.05, sk)
        else:
            self._draw_box(ax, 5.6, 2.8, "x = x + a", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 5.6, 3.5, 5.6, 3.15, sk)
            self._draw_box(ax, 5.6, 1.7, "x = x * b", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 5.6, 2.4, 5.6, 2.05, sk)
        # No
        self._draw_arrow(ax, 1.9, 3.9, 1.0, 3.9, sk, "No")
        if not (val > threshold):
            self._draw_box(ax, 1.0, 2.8, f"x = x - {v_sub}", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 1.0, 3.5, 1.0, 3.15, sk)
            self._draw_box(ax, 1.0, 1.7, f"x = x + {v_add2}", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 1.0, 2.4, 1.0, 2.05, sk)
        else:
            self._draw_box(ax, 1.0, 2.8, "x = x - c", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 1.0, 3.5, 1.0, 3.15, sk)
            self._draw_box(ax, 1.0, 1.7, "x = x + d", sk, w=2.0, h=0.75)
            self._draw_arrow(ax, 1.0, 2.4, 1.0, 2.05, sk)
        ax.set_xlim(-0.4, 7.2); ax.set_ylim(0.5, 5.7); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        sidx = (self.seed or 0) % 16
        q = _Q_TWO_BRANCH[sidx].format(val=val)
        return q, str(result), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _three_decision_output(self, rng, style_rng):
        x = rng.randint(1, 30)
        t1 = rng.randint(5, 15)
        t2 = rng.randint(10, 25)
        t3 = rng.randint(1, 20)
        if x > t1:
            if x > t2:
                result = x * 2
            else:
                result = x + 10
        else:
            if x > t3:
                result = x - 5
            else:
                result = x * 3

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(8 * _fs, 7.5 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 4, 7.2, f"x = {x}", sk, shape="oval", w=1.8, h=0.7)
        self._draw_arrow(ax, 4, 6.85, 4, 6.25, sk)
        self._draw_box(ax, 4, 5.9, f"x > {t1}?", sk, shape="diamond", w=2.2, h=0.9)
        self._draw_arrow(ax, 5.2, 5.9, 6.2, 5.9, sk, "Yes")
        self._draw_box(ax, 6.2, 4.5, f"x > {t2}?", sk, shape="diamond", w=2.2, h=0.9)
        self._draw_arrow(ax, 6.2, 5.5, 6.2, 5.0, sk)
        self._draw_arrow(ax, 7.3, 4.5, 7.7, 4.5, sk, "Yes")
        self._draw_box(ax, 7.7, 3.2, "x = x * 2", sk, w=1.6, h=0.75)
        self._draw_arrow(ax, 7.7, 4.1, 7.7, 3.57, sk)
        self._draw_box(ax, 7.7, 2.0, "Output", sk, shape="oval", w=1.5, h=0.75)
        self._draw_arrow(ax, 7.7, 2.85, 7.7, 2.38, sk)
        self._draw_arrow(ax, 5.1, 4.5, 4.7, 4.5, sk, "No")
        self._draw_box(ax, 4.7, 3.2, "x = x + 10", sk, w=1.6, h=0.75)
        self._draw_arrow(ax, 4.7, 4.1, 4.7, 3.57, sk)
        self._draw_box(ax, 4.7, 2.0, "Output", sk, shape="oval", w=1.5, h=0.75)
        self._draw_arrow(ax, 4.7, 2.85, 4.7, 2.38, sk)
        self._draw_arrow(ax, 2.8, 5.9, 1.8, 5.9, sk, "No")
        self._draw_box(ax, 1.8, 4.5, f"x > {t3}?", sk, shape="diamond", w=2.2, h=0.9)
        self._draw_arrow(ax, 1.8, 5.5, 1.8, 5.0, sk)
        self._draw_arrow(ax, 2.9, 4.5, 3.3, 4.5, sk, "Yes")
        self._draw_box(ax, 3.3, 3.2, "x = x - 5", sk, w=1.6, h=0.75)
        self._draw_arrow(ax, 3.3, 4.1, 3.3, 3.57, sk)
        self._draw_box(ax, 3.3, 2.0, "Output", sk, shape="oval", w=1.5, h=0.75)
        self._draw_arrow(ax, 3.3, 2.85, 3.3, 2.38, sk)
        self._draw_arrow(ax, 0.7, 4.5, 0.3, 4.5, sk, "No")
        self._draw_box(ax, 0.3, 3.2, "x = x * 3", sk, w=1.6, h=0.75)
        self._draw_arrow(ax, 0.3, 4.1, 0.3, 3.57, sk)
        self._draw_box(ax, 0.3, 2.0, "Output", sk, shape="oval", w=1.5, h=0.75)
        self._draw_arrow(ax, 0.3, 2.85, 0.3, 2.38, sk)
        ax.set_xlim(-0.7, 9); ax.set_ylim(1.4, 8); ax.axis("off")
        ax.set_title(sk["title"], fontsize=12, pad=4)
        sidx = (self.seed or 0) % 16
        q = _Q_THREE_DECISION[sidx].format(x=x)
        return q, str(result), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _reverse_engineer_input(self, rng, style_rng):
        # result = (x + a) * b
        target = rng.randint(14, 80)
        add_val = rng.randint(2, 10)
        mul_val = rng.randint(2, 5)
        if target % mul_val != 0:
            target = mul_val * rng.randint(3, 15)
        x = target // mul_val - add_val

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(4.6 * _fs, 5.2 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 2, 4.6, "x = ?", sk, shape="oval", w=1.8, h=0.75)
        self._draw_arrow(ax, 2, 4.22, 2, 3.57, sk)
        self._draw_box(ax, 2, 3.2, f"x = x + {add_val}", sk, w=2.2, h=0.8)
        self._draw_arrow(ax, 2, 2.82, 2, 2.25, sk)
        self._draw_box(ax, 2, 1.9, f"x = x * {mul_val}", sk, w=2.2, h=0.8)
        self._draw_arrow(ax, 2, 1.52, 2, 0.95, sk)
        self._draw_box(ax, 2, 0.6, f"Output: {target}", sk, shape="oval", w=2.0, h=0.8)
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(-0.2, 5.2); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        sidx = (self.seed or 0) % 16
        q = _Q_REVERSE[sidx].format(target=target)
        return q, str(x), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _loop_iteration_count(self, rng, style_rng):
        start = rng.randint(1, 6)
        inc = rng.randint(1, 3)
        limit = rng.randint(12, 30)
        count = 0; val = start
        while val < limit:
            val += inc; count += 1
        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(5.6 * _fs, 6 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 2.5, 5.4, self._maybe_hide_input(f"x = {start}"), sk, shape="oval", w=1.8, h=0.75)
        self._draw_arrow(ax, 2.5, 5.05, 2.5, 4.45, sk)
        self._draw_box(ax, 2.5, 4.1, f"x < {limit}?", sk, shape="diamond", w=2.2, h=0.9)
        self._draw_arrow(ax, 3.7, 4.1, 4.8, 4.1, sk, "Yes")
        self._draw_box(ax, 4.8, 2.9, f"x = x + {inc}", sk, w=2.0, h=0.8)
        self._draw_arrow(ax, 4.8, 3.7, 4.8, 3.27, sk)
        ax.annotate("", xy=(2.5, 4.45), xytext=(4.8, 2.55),
                    arrowprops=dict(arrowstyle="->", color=sk["arrow"], lw=1.5,
                                      connectionstyle="arc3,rad=-0.3"))
        self._draw_arrow(ax, 1.3, 4.1, 0.4, 4.1, sk, "No")
        self._draw_box(ax, 0.4, 2.9, "Done", sk, shape="oval", w=1.6, h=0.75)
        self._draw_arrow(ax, 0.4, 3.7, 0.4, 3.28, sk)
        self._add_decoy_branches(ax, sk, rng, (6.2, 7.5, 3.0, 5.5))
        ax.set_xlim(-0.4, 7.8); ax.set_ylim(2.0, 6); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        sidx = (self.seed or 0) % 16
        if getattr(self, "_flow_cfg", {}).get("hide_input_in_box"):
            q = _Q_LOOP_HIDDEN[sidx]
        else:
            q = _Q_LOOP_VISIBLE[sidx].format(start=start, inc=inc, limit=limit)
        return q, str(count), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _multi_variable_trace(self, rng, style_rng):
        x = rng.randint(1, 8); y = rng.randint(1, 8)
        ops_pool = ["swap", "add_to_x", "multiply_y", "subtract_from_y"]
        ops = rng.sample(ops_pool, 2)
        rx, ry = x, y
        op_texts = []
        for op in ops:
            if op == "swap":
                rx, ry = ry, rx; op_texts.append("swap x, y")
            elif op == "add_to_x":
                v = rng.randint(2, 5); rx += v; op_texts.append(f"x = x + {v}")
            elif op == "multiply_y":
                v = rng.randint(2, 4); ry *= v; op_texts.append(f"y = y * {v}")
            elif op == "subtract_from_y":
                v = rng.randint(2, 5); ry -= v; op_texts.append(f"y = y - {v}")
        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(4.4 * _fs, 5.2 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        y_pos = 4.6
        self._draw_box(ax, 2, y_pos, f"x={x}, y={y}", sk, shape="oval", w=2.6, h=0.75)
        prev_y = y_pos
        for t in op_texts:
            y_pos -= 1.1
            self._draw_box(ax, 2, y_pos, t, sk, w=2.5, h=0.75)
            self._draw_arrow(ax, 2, prev_y - 0.4, 2, y_pos + 0.4, sk)
            prev_y = y_pos
        y_pos -= 1.1
        self._draw_box(ax, 2, y_pos, "Output x + y", sk, shape="oval", w=2.5, h=0.75)
        self._draw_arrow(ax, 2, prev_y - 0.4, 2, y_pos + 0.4, sk)
        self._add_decoy_branches(ax, sk, rng, (4.8, 6.3, y_pos + 0.5, 4.5))
        ax.set_xlim(-0.4, 6.8); ax.set_ylim(y_pos - 0.7, 5.2); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        total = rx + ry
        sidx = (self.seed or 0) % 16
        if getattr(self, "_flow_cfg", {}).get("hide_input_in_box"):
            q = _Q_MULTI_HIDDEN[sidx]
        else:
            q = _Q_MULTI_VISIBLE[sidx].format(x=x, y=y)
        return q, str(total), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _deep_multi_variable_trace(self, rng, style_rng):
        x = rng.randint(1, 8); y = rng.randint(1, 8); z = rng.randint(1, 6)
        # L8-L9 hardening 2026-04-17: add a conditional IF-branch op so the
        # student must read the decision diamond correctly. Also allow 5 ops
        # (up from 4) and add richer ops.
        hide_input = getattr(self, "_flow_cfg", {}).get("hide_input_in_box")
        use_conditional = getattr(self, "_flow_cfg", {}).get("add_decoy_branches")
        ops_pool = ["add_z_to_x", "multiply_y_by_z", "swap_xy",
                     "subtract_z_from_y", "x_mod_z", "double_x",
                     "sub_y_from_x"]
        n_ops = 5 if use_conditional else 4
        ops = rng.sample(ops_pool, min(n_ops, len(ops_pool)))
        # Decide where the conditional step sits (if enabled)
        cond_idx = rng.randint(1, len(ops) - 1) if use_conditional else -1
        cond_threshold = rng.randint(3, 10)
        # Conditional op: "if x > {thr}: y = y + z  else: y = y - z"
        rx, ry, rz = x, y, z
        op_texts = []
        for i, op in enumerate(ops):
            if i == cond_idx:
                # insert conditional
                if rx > cond_threshold:
                    ry += rz
                else:
                    ry -= rz
                op_texts.append(f"if x > {cond_threshold}: y=y+z else: y=y-z")
                continue
            if op == "add_z_to_x":
                rx += rz; op_texts.append("x = x + z")
            elif op == "multiply_y_by_z":
                ry *= rz; op_texts.append("y = y * z")
            elif op == "swap_xy":
                rx, ry = ry, rx; op_texts.append("swap x, y")
            elif op == "subtract_z_from_y":
                ry -= rz; op_texts.append("y = y - z")
            elif op == "x_mod_z":
                if rz > 0:
                    rx = rx % rz
                op_texts.append("x = x mod z")
            elif op == "double_x":
                rx *= 2; op_texts.append("x = 2 * x")
            elif op == "sub_y_from_x":
                rx -= ry; op_texts.append("x = x - y")

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(5.5 * _fs, 7.5 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        y_pos = 6.2
        # Input oval — show values (never hide; the values live only on image).
        self._draw_box(ax, 2.5, y_pos, f"x={x}, y={y}, z={z}",
                       sk, shape="oval", w=3.0, h=0.75)
        prev_y = y_pos
        for i, t in enumerate(op_texts):
            y_pos -= 1.0
            # Draw conditional as diamond
            if i == cond_idx:
                self._draw_box(ax, 2.5, y_pos, t, sk,
                               shape="diamond", w=3.6, h=1.0)
            else:
                self._draw_box(ax, 2.5, y_pos, t, sk, w=2.9, h=0.75)
            self._draw_arrow(ax, 2.5, prev_y - 0.4, 2.5, y_pos + 0.4, sk)
            prev_y = y_pos
        y_pos -= 1.0
        self._draw_box(ax, 2.5, y_pos, "Output x * y", sk,
                       shape="oval", w=2.9, h=0.75)
        self._draw_arrow(ax, 2.5, prev_y - 0.4, 2.5, y_pos + 0.4, sk)
        self._add_decoy_branches(ax, sk, rng, (5.6, 7.2, y_pos + 0.5, 5.5))
        ax.set_xlim(-0.4, 7.8); ax.set_ylim(y_pos - 0.7, 7.0); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        total = rx * ry
        sidx = (self.seed or 0) % 16
        if hide_input:
            q = _Q_DEEP_HIDDEN[sidx]
        else:
            q = _Q_DEEP_VISIBLE[sidx].format(x=x, y=y, z=z)
        return q, str(total), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _nested_loop_count(self, rng, style_rng):
        n = rng.randint(4, 10)
        total = n * (n + 1) // 2

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(6 * _fs, 6 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 3, 5.4, self._maybe_hide_input("sum=0, i=1"), sk, shape="oval", w=2.5, h=0.75)
        self._draw_arrow(ax, 3, 5.05, 3, 4.45, sk)
        self._draw_box(ax, 3, 4.1, f"i <= {n}?", sk, shape="diamond", w=2.2, h=0.9)
        self._draw_arrow(ax, 4.3, 4.1, 5.2, 4.1, sk, "Yes")
        self._draw_box(ax, 5.2, 2.9, "sum = sum + i", sk, w=2.2, h=0.8)
        self._draw_arrow(ax, 5.2, 3.7, 5.2, 3.27, sk)
        self._draw_box(ax, 5.2, 1.7, "i = i + 1", sk, w=2.0, h=0.8)
        self._draw_arrow(ax, 5.2, 2.5, 5.2, 2.05, sk)
        ax.annotate("", xy=(3, 4.45), xytext=(5.2, 1.35),
                    arrowprops=dict(arrowstyle="->", color=sk["arrow"], lw=1.5,
                                     connectionstyle="arc3,rad=-0.3"))
        self._draw_arrow(ax, 1.7, 4.1, 0.8, 4.1, sk, "No")
        self._draw_box(ax, 0.8, 2.9, "Output sum", sk, shape="oval", w=1.8, h=0.75)
        self._draw_arrow(ax, 0.8, 3.7, 0.8, 3.27, sk)
        self._add_decoy_branches(ax, sk, rng, (-0.3, 1.7, 1.0, 2.6))
        ax.set_xlim(-0.4, 6.7); ax.set_ylim(0.8, 6); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        sidx = (self.seed or 0) % 16
        if getattr(self, "_flow_cfg", {}).get("hide_input_in_box"):
            q = _Q_NESTED_HIDDEN[sidx]
        else:
            q = _Q_NESTED_VISIBLE[sidx].format(n=n)
        return q, str(total), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _deep_program_trace(self, rng, style_rng):
        """L9 iter-3: 9-op program with OUTER LOOP + nested IF + 3 mutable
        variables x/y/z. Model must simulate the whole program faithfully.

        Program structure (rendered as a long flowchart):

          1. x = X0, y = Y0, z = Z0, i = 0, sum = 0
          2. while i < N:
               3. if x > y: x = x + z
                  else:    y = y + z
               4. z = z - 1 (floor at 0)
               5. if x % 2 == 0: sum = sum + x
                  else:          sum = sum + y
               6. if z == 0: break
               7. i = i + 1
          8. output sum + x - y
        """
        # Pick starting values so the program terminates in ≤ 10 iterations
        # but still has non-trivial state evolution.
        X0 = rng.randint(3, 9)
        Y0 = rng.randint(3, 9)
        Z0 = rng.randint(4, 8)   # so z will decrement several times
        N = rng.randint(6, 9)
        # Simulate
        x, y, z = X0, Y0, Z0
        i = 0
        s = 0
        steps = []  # not used for rendering, but helpful for debug
        iter_done = 0
        for _ in range(20):  # safety cap
            if iter_done >= N:
                break
            # 3
            if x > y:
                x = x + z
            else:
                y = y + z
            # 4
            if z > 0:
                z -= 1
            # 5
            if x % 2 == 0:
                s += x
            else:
                s += y
            iter_done += 1
            # 6
            if z == 0:
                break
            # 7 (implicit): i+=1
        result = s + x - y

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        # BUGFIX 2026-04-24: widen figure + widen START oval so long text like
        # "x=7, y=9, z=5, i=0, sum=0" fits without clipping at some seeds.
        fig, ax = plt.subplots(figsize=(8.0 * _fs, 11.5 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        yp = 13.0
        self._draw_box(ax, 3.5, yp,
                       f"x={X0}, y={Y0}, z={Z0}, i=0, sum=0",
                       sk, shape="oval", w=6.2, h=0.95)
        yp -= 0.95
        self._draw_arrow(ax, 3.5, yp + 0.45, 3.5, yp, sk)
        self._draw_box(ax, 3.5, yp, f"i < {N}?", sk,
                       shape="diamond", w=2.4, h=0.95)
        # Yes-branch indicator right into the loop body
        self._draw_arrow(ax, 4.8, yp, 5.8, yp, sk, "Yes")
        # No: exit right (to the output block way below)
        self._draw_arrow(ax, 2.2, yp, 1.0, yp, sk, "No")

        # Body column on the right
        body_x = 5.8
        yp -= 0.95
        self._draw_box(ax, body_x, yp, "if x > y: x = x + z  else: y = y + z",
                       sk, shape="diamond", w=4.8, h=1.0)
        yp -= 1.0
        self._draw_box(ax, body_x, yp, "z = max(z - 1, 0)", sk, w=3.4, h=0.8)
        yp -= 0.9
        self._draw_box(ax, body_x, yp,
                       "if x%2==0: sum=sum+x  else: sum=sum+y",
                       sk, shape="diamond", w=5.0, h=1.0)
        yp -= 1.0
        self._draw_box(ax, body_x, yp, "z == 0?", sk,
                       shape="diamond", w=2.4, h=0.9)
        self._draw_arrow(ax, body_x + 1.3, yp, body_x + 2.3, yp, sk,
                         "Yes → break")
        yp -= 0.9
        self._draw_box(ax, body_x, yp, "i = i + 1", sk, w=2.0, h=0.8)
        # loop back arrow (curved) from here to the i<N diamond
        ax.annotate("", xy=(3.5, 12.05 - 1.9),  # approximate anchor
                    xytext=(body_x, yp - 0.5),
                    arrowprops=dict(arrowstyle="->", color=sk["arrow"],
                                    lw=1.3,
                                    connectionstyle="arc3,rad=-0.35"))

        # Output box on the left
        yp_out = 1.0
        # BUGFIX 2026-04-24: shifted x=1.0 -> x=1.5 and widened w to avoid
        # left-side clipping of "output sum + x - y" oval.
        self._draw_box(ax, 1.5, yp_out,
                       "output sum + x - y", sk, shape="oval",
                       w=4.2, h=0.85)
        # Connect: No (from i<N) comes out to the left then down.
        ax.annotate("", xy=(1.5, yp_out + 0.5),
                    xytext=(1.5, 12.05 - 1.0 + 0.2),
                    arrowprops=dict(arrowstyle="->", color=sk["arrow"],
                                    lw=1.3))
        # Break arrow ends at the output too (long curved)
        ax.annotate("", xy=(1.5, yp_out + 0.5),
                    xytext=(body_x + 2.3, 12.05 - 5.7),
                    arrowprops=dict(arrowstyle="->", color=sk["arrow"],
                                    lw=1.3,
                                    connectionstyle="arc3,rad=0.3"))

        self._add_decoy_branches(ax, sk, rng, (-0.2, 0.8, 2.0, 8.0))
        # BUGFIX 2026-04-24: widen xlim so START oval (x=3.5, w=6.2 -> spans
        # 0.4..6.6) and output oval (x=1.5, w=4.2 -> spans -0.6..3.6) do not
        # clip. Previous xlim=-0.5 cut the output oval's left edge and some
        # wider START oval text (e.g. seed=5 "x=7...").
        ax.set_xlim(-1.0, 12.2)
        ax.set_ylim(-0.2, 14.2)
        ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)

        sidx = (self.seed or 0) % 16
        q = _Q_DEEP_PROG[sidx]
        return q, str(result), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))

    def _conditional_accumulator(self, rng, style_rng):
        limit = rng.randint(10, 25)
        start = rng.randint(1, 4)
        step = rng.randint(1, 2)
        total = 0; x = start
        while x <= limit:
            if x % 2 == 1:
                total += x
            x += step

        sk = self._style_kit(style_rng)
        _fs = sk.get("figsize_scale", 1.0)
        fig, ax = plt.subplots(figsize=(7.2 * _fs, 7 * _fs))
        fig.patch.set_facecolor(sk["bg"])
        self._draw_box(ax, 3.5, 6.5, self._maybe_hide_input(f"x={start}, sum=0"), sk, shape="oval", w=2.5, h=0.75)
        self._draw_arrow(ax, 3.5, 6.13, 3.5, 5.57, sk)
        self._draw_box(ax, 3.5, 5.2, f"x <= {limit}?", sk, shape="diamond", w=2.2, h=0.9)
        self._draw_arrow(ax, 4.6, 5.2, 5.5, 5.2, sk, "Yes")
        self._draw_box(ax, 5.5, 4.0, "x odd?", sk, shape="diamond", w=2.0, h=0.75)
        self._draw_arrow(ax, 5.5, 4.8, 5.5, 4.38, sk)
        self._draw_arrow(ax, 6.5, 4.0, 7.0, 4.0, sk, "Yes")
        self._draw_box(ax, 7.0, 2.8, "sum += x", sk, w=1.5, h=0.75)
        self._draw_arrow(ax, 7.0, 3.65, 7.0, 3.18, sk)
        self._draw_arrow(ax, 4.5, 4.0, 4.0, 4.0, sk, "No")
        self._draw_box(ax, 5.5, 1.6, f"x = x + {step}", sk, w=2.0, h=0.75)
        self._draw_arrow(ax, 5.5, 2.4, 5.5, 1.98, sk)
        ax.annotate("", xy=(3.5, 5.57), xytext=(5.5, 1.25),
                    arrowprops=dict(arrowstyle="->", color=sk["arrow"], lw=1.5,
                                     connectionstyle="arc3,rad=-0.4"))
        self._draw_arrow(ax, 2.4, 5.2, 1.5, 5.2, sk, "No")
        self._draw_box(ax, 1.5, 4.0, "Output sum", sk, shape="oval", w=1.8, h=0.75)
        self._draw_arrow(ax, 1.5, 4.8, 1.5, 4.38, sk)
        self._add_decoy_branches(ax, sk, rng, (-0.3, 1.1, 1.0, 3.0))
        ax.set_xlim(-0.5, 8.5); ax.set_ylim(0.8, 7.2); ax.axis("off")
        ax.set_title(sk["title"], fontsize=11, pad=4)
        sidx = (self.seed or 0) % 16
        if getattr(self, "_flow_cfg", {}).get("hide_input_in_box"):
            q = _Q_COND_HIDDEN[sidx]
        else:
            q = _Q_COND_VISIBLE[sidx].format(start=start, step=step, limit=limit)
        return q, str(total), self.fig_to_pil(fig, dpi=sk.get("dpi", 110))
