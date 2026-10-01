# Case 5: why does a missing premise elicit a number?

This exploratory diagnostic follows the first case-5 pilot. It is separate
from the original case-5 pairs and is **not** a frozen final benchmark. Generate
fresh numerical instances for evaluation after inspecting a small pilot. All
three quantities in each base problem are distinct. The same base is rendered
in nine variants; each variant is sent as an independent, single-turn chat.

| Test | Variants | Interpretation |
| --- | --- | --- |
| 1. Missing-number location | Complete; missing initial state; missing received amount; missing given-away amount | A numerical-guess rate concentrated on the first missing location suggests an initial-state default. Guesses across locations suggest a broader sufficiency problem. |
| 2. Implicit/explicit unknown | Missing initial state; explicitly unspecified initial state; zero initial state; ask for net change despite unknown initial state | Separates omission sensitivity from a blanket response to words such as “unspecified.” The zero and net-change controls both have numerical answers. |
| 3. Solve/answerability | Complete and missing-initial versions, each asked as an ordinary solve question and as a unique-answer yes/no question | Correctly answering “no” while giving an unsupported number on the corresponding solve prompt is the target dissociation. |

The missing initial condition omits the entire opening fact. The missing
received/given-away conditions retain the action with “some” instead of its
amount. That difference in wording is a limitation when attributing a location
effect. The explicit-unknown variant deliberately tests that wording change.
The net-change question asks for the signed change, which is `received − given`
regardless of initial state.

The solve prompt accepts a number, “cannot be determined,” or a simple
expression in `x`, where `x` stands for the unknown amount. The answerability
prompt requests “yes” or “no” plus an explanation. Both models use thinking;
Qwen has `enable_thinking=True` and OLMo uses native thinking. The requested
generation and context limits are 4,096 and 8,192 tokens on one A100 40 GB
GPU at `gpu_memory_utilization=0.85`. These are **pilot settings**, not a
verified hardware maximum. If truncation remains high, try 8,192 generated
tokens with a 16,384-token context on a small set before a larger run.

The scorer records correct numeric answers, correct recognition of
underdetermination, correct symbolic expressions, explicit conditional
answers, unsupported numerical answers, other errors, and truncations.
Conditional answers are reported separately and are not counted as exact
correct answers. `zero_equivalent_guess` flags an unsupported number equal to
the result obtained by setting the missing quantity to zero; it does not prove
the model actually made that assumption. All raw responses and model/prompt
settings are saved, so ambiguous or unparsed responses can be audited manually.
These categories describe behavior; they do not establish a model's internal
causal mechanism.
