"""Task prompts (T0-T4), elicitation prompts (ProbInt, Pos), and their parsers.

Elicitation instructions quote Yang, Muandet et al. (UAI 2026) Prompt 2 (ProbInt) and Prompt 4
(Pos). Our additions: the fixed numeric grid replaces model-generated candidates, answers may be
given as ranges, omitted answers count as zero, and Pos has no "different answer" option because
the grid is exhaustive.
"""

from __future__ import annotations

import json
import re

from study.generate_shortcut_cases import UNDETERMINED
from study.run_case5_assumption_pilot import INSTRUCTION
from study.run_case5_confirmatory import SYSTEM_CHECK, SYSTEM_SOLVE
from study.score_case5_confirmatory import ABSTAIN, FINAL, _affine
from study.uc_items import GRID, GRID_MAX, TAIL

PARSER_VERSION = "uc-v2"
TASKS = ("T0", "T1", "T2", "T3", "T4")
FORMATS = ("probint", "pos")
ARMS = ("direct", "two_stage")
SYSTEM_PLAIN = "Solve the problem. End with one line: Final answer: followed by your answer."

_GRID_TEXT = (f"Consider every possible answer to the question above: each whole number from 0 to "
              f"{GRID_MAX}, and \"{TAIL}\".")
_FORMAT_TEXT = ("End with one line: Final answer: followed by a JSON object. Each key is a "
                "possible answer: a single number such as \"7\", an inclusive range such as "
                "\"12-20\" (the value applies to each number in the range), or "
                f"\"{TAIL}\". A single number overrides a range that contains it. Every answer "
                "you leave out gets {zero}.")
ELICIT = {
    "probint": "\n\n".join([
        _GRID_TEXT,
        "For each possible answer, provide a lower and upper probability (each between 0.0 and "
        "1.0) indicating how likely the answer is correct. Interpret the probabilities as "
        "follows:\n"
        "- Lower Probability: the smallest probability you consider plausible that the answer "
        "is correct.\n"
        "- Upper Probability: the largest probability you consider defensible that the answer "
        "is correct.\n"
        "The sum of all lower probabilities across all answers must not exceed 1.0.",
        _FORMAT_TEXT.format(zero="[0.0, 0.0]") + " Write each value as [lower, upper]."]),
    "pos": "\n\n".join([
        _GRID_TEXT,
        "For each possible answer, provide a possibility score which captures how plausible the "
        "answer correctly answers the question. The possibility should be between 0.0 and 1.0. "
        "A possibility score of 1.0 means \"fully plausible,\" and 0.0 means \"impossible.\"",
        _FORMAT_TEXT.format(zero="0.0")]),
}


def _merge_system(messages: list[dict], system_in_user: bool) -> list[dict]:
    if not system_in_user or messages[0]["role"] != "system":
        return messages
    system, first, *rest = messages
    return [{"role": "user", "content": f"{system['content']}\n\n{first['content']}"}, *rest]


def task_messages(problem: dict, task: str, system_in_user: bool = False) -> list[dict]:
    facts, name, noun = problem["facts"], problem["name"], problem["noun"]
    if task == "T0":
        system, user = SYSTEM_PLAIN, f"{facts} {problem['question']}"
    elif task == "T1":
        system, user = SYSTEM_SOLVE, f"{facts} {problem['question']}"
    elif task == "T2":
        system, user = SYSTEM_SOLVE, f"{facts} {problem['question']}\n\n{INSTRUCTION}"
    elif task == "T3":
        system, user = SYSTEM_PLAIN, f"{facts} What can be concluded about how many {noun} {name} has now?"
    elif task == "T4":
        system = SYSTEM_CHECK
        user = f"{facts} Does the information uniquely determine how many {noun} {name} has now?"
    else:
        raise ValueError(task)
    return _merge_system([{"role": "system", "content": system},
                          {"role": "user", "content": user}], system_in_user)


def direct_messages(problem: dict, fmt: str) -> list[dict]:
    return [{"role": "user", "content": f"{problem['facts']} {problem['question']}\n\n{ELICIT[fmt]}"}]


# The task's system prompt asks for a number; without this the two instructions conflict.
OVERRIDE = ("For this request only, ignore the earlier instruction about the final answer "
            "format and follow the format below.")


def two_stage_messages(problem: dict, task: str, answer: str, fmt: str,
                       system_in_user: bool = False) -> list[dict]:
    return [*task_messages(problem, task, system_in_user),
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"{OVERRIDE}\n\n{ELICIT[fmt]}"}]


def visible(raw: str, mode: str) -> str | None:
    """Text after the thinking block; None if a thinking-mode response never closed it."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[-1].strip()
    return None if mode == "thinking" else raw.strip()


def _final_line(text: str) -> str | None:
    matches = list(FINAL.finditer(text))
    if not matches:
        return None
    line = text[matches[-1].end():].split("\n", 1)[0].replace("**", "").strip().strip("$ ")
    line = re.sub(r"^\\boxed\{(.*)\}\.?$", r"\1", line).strip()
    line = re.sub(r"^\\text\{(.*)\}$", r"\1", line).strip()
    return line.rstrip(".").strip() or None


_INT = r"(\d{1,3})"
_SET_PATTERNS = [
    (re.compile(rf"(?i)^(?:between\s+|from\s+)?{_INT}\s*(?:and|to|-|–)\s*{_INT}(?:\s+\w+)?(?:\s*\(?inclusive\)?)?$"),
     lambda a, b: list(range(int(a), int(b) + 1))),
    (re.compile(rf"(?i)^(?:at\s+least\s+{_INT}|{_INT}\s+or\s+more|(?:≥|>=)\s*{_INT})(?:\s+\w+)?$"),
     lambda *g: ("from", int(next(x for x in g if x)))),
    (re.compile(rf"(?i)^more\s+than\s+{_INT}(?:\s+\w+)?$"), lambda a: ("from", int(a) + 1)),
]


def _labels(values) -> list[str]:
    if isinstance(values, tuple):
        return [str(v) for v in range(values[1], GRID_MAX + 1)] + [TAIL]
    return sorted({str(v) if v <= GRID_MAX else TAIL for v in values}, key=GRID.index)


def classify_answer(problem: dict, task: str, raw: str, finish_reason: str | None,
                    mode: str) -> dict:
    """Stage-1 outcome. 'single' commits to one value; a stated set is compared to the true set."""
    if finish_reason == "length":
        return {"outcome": "truncated"}
    text = visible(raw, mode)
    line = _final_line(text) if text else None
    if line is None:
        return {"outcome": "unparseable"}
    true = problem["true_set"]
    if task == "T4":
        match = re.match(r"(?i)^(yes|no)\b", line)
        if not match:
            return {"outcome": "unparseable"}
        parsed = match.group(1).lower()
        return {"outcome": "answerability", "parsed": parsed,
                "correct": parsed == ("yes" if len(true) == 1 else "no")}
    if ABSTAIN.search(line) and not re.search(r"\d", line):
        return {"outcome": "abstain", "parsed": UNDETERMINED, "correct": len(true) > 1}
    affine = _affine(line)
    if affine is not None and affine[0] != 0:
        return {"outcome": "expression", "parsed": line,
                "correct": problem["content"] == "unconstrained" and affine == (1, problem["zero_default"])}
    number = re.fullmatch(r"([+-]?\d+)(?:\s+[A-Za-z]+)?", line.replace(",", ""))
    if number:
        value = int(number.group(1))
        label = str(value) if 0 <= value <= GRID_MAX else (TAIL if value > GRID_MAX else None)
        return {"outcome": "single", "parsed": value, "label": label,
                "in_true_set": label in true, "correct": len(true) == 1 and label in true,
                "zero_default": value == problem["zero_default"]}
    words = re.sub(r"\s+(?:or|and)\s+", ", ", line)
    parts = [p.strip() for p in words.split(",") if p.strip()]
    if len(parts) > 1 and all(re.fullmatch(r"\d{1,3}", p) for p in parts):
        stated = _labels(int(p) for p in parts)
        return {"outcome": "set", "parsed": stated, "correct": stated == true}
    for pattern, build in _SET_PATTERNS:
        match = pattern.fullmatch(line)
        if match:
            stated = _labels(build(*match.groups()))
            return {"outcome": "set", "parsed": stated, "correct": stated == true}
    return {"outcome": "unparseable", "parsed": line}


def _json_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    value = json.loads(text[start:end + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    break
        start = text.find("{", start + 1)
    return None


def _key_labels(key: str) -> list[str] | None:
    key = key.strip().lower().replace("–", "-")
    if key in (TAIL, f">{GRID_MAX}", f"> {GRID_MAX}"):
        return [TAIL]
    if re.fullmatch(r"\d{1,3}", key):
        return [key] if int(key) <= GRID_MAX else None
    match = re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", key)
    if match and int(match.group(1)) <= int(match.group(2)) <= GRID_MAX:
        return [str(v) for v in range(int(match.group(1)), int(match.group(2)) + 1)]
    return None


def parse_profile(fmt: str, raw: str, finish_reason: str | None, mode: str) -> dict:
    """Parse and validate an elicited profile over the full grid (Yang et al.'s verifier checks)."""
    if finish_reason == "length":
        return {"valid": False, "error": "truncated"}
    text = visible(raw, mode)
    if text is None:
        return {"valid": False, "error": "no_closed_thinking"}
    matches = list(FINAL.finditer(text))
    obj = _json_object(text[matches[-1].end():]) if matches else None
    if obj is None:  # e.g. the JSON comes before a trailing "Final answer: 5"
        blocks = [m.start() for m in re.finditer(r"\{", text)]
        obj = next((o for o in (_json_object(text[i:]) for i in reversed(blocks)) if o), None)
    if obj is None:
        return {"valid": False, "error": "no_json"}
    zero = [0.0, 0.0] if fmt == "probint" else 0.0
    profile = {y: zero for y in GRID}
    # A range whose value is an object of its own single answers is flattened.
    flat = {}
    for key, value in obj.items():
        if isinstance(value, dict) and value and all(
                _key_labels(str(k)) and set(_key_labels(str(k))) <= set(_key_labels(str(key)) or [])
                for k in value):
            flat.update(value)
        else:
            flat[key] = value
    singles, ranges = [], []
    for key, value in flat.items():
        labels = _key_labels(str(key))
        if labels is None:
            return {"valid": False, "error": f"bad_key:{key}"}
        (singles if len(labels) == 1 else ranges).append((labels, value))
    for labels, value in ranges + singles:  # singles applied last, so they override ranges
        if fmt == "probint":
            if (not isinstance(value, list) or len(value) != 2 or
                    not all(isinstance(v, (int, float)) for v in value)):
                return {"valid": False, "error": f"bad_value:{value}"}
            value = [float(value[0]), float(value[1])]
            if not 0 <= value[0] <= value[1] <= 1:
                return {"valid": False, "error": f"bad_interval:{value}"}
        else:
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                return {"valid": False, "error": f"bad_value:{value}"}
            value = float(value)
        for y in labels:
            profile[y] = value
    if fmt == "probint":
        lower = sum(v[0] for v in profile.values())
        upper = sum(v[1] for v in profile.values())
        if lower > 1 + 1e-6:
            return {"valid": False, "error": "lower_sum_above_1", "profile": profile}
        if upper < 1 - 1e-6:
            return {"valid": False, "error": "upper_sum_below_1", "profile": profile}
    elif max(profile.values()) <= 0:
        return {"valid": False, "error": "all_zero", "profile": profile}
    return {"valid": True, "profile": profile}
