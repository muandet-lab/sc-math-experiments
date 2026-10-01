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

## Assumption-instruction follow-up

The final exploratory pilot reuses the same ten bases and only three existing
solve variants: omitted initial amount, complete problem, and net change with
an explicitly unknown initial amount. Each is sent in a fresh single-turn
conversation with the same additional system instruction:

> Use only the stated facts. Do not assign values to quantities whose values
> are not given. If the requested quantity is not uniquely determined, state
> that.

The input questions, answer format, model settings, and sampling seed match the
earlier pilot. Each model therefore produces 30 additional responses. Compare
each response with the same base and variant in its saved baseline. Fewer
unsupported numbers on omitted-start questions with intact control accuracy
suggests a task-cue-correctable failure. Persisting guesses despite the
instruction suggests a more robust shortcut. Incorrect abstention on complete
or net-change controls indicates over-abstention. These outcomes cannot by
themselves establish the internal cause or training origin. One sampled
response per condition makes this an exploratory pilot.

## Incremental instruction-placement Stage 1

The next exploratory step reuses the original ten bases again. It adds only
13 previously unobserved condition/placement cells per base (130 responses per
model). The two new conditions are “has a box of [objects]” without a count
and “has as many [objects] as Tom” without Tom's count. The vague-transfer
condition is consistently “gives away some” for both models. Existing
baseline and system-instruction results fill the other cells; explicit-unknown
under instruction is omitted because both models were 10/10 at baseline.

The user-placement treatment appends the exact same instruction quoted above
to the end of the user message. The system-placement treatment appends it to
the existing solve system prompt. This is a within-base exploratory comparison,
not an independent confirmatory sample: old and new generations are single
draws from stochastic models, and a condition that is already at ceiling
cannot demonstrate instruction efficacy. Check that at least one marked
condition has baseline failures before using it as a positive control.
