# Case-5 OLMo follow-up at 8,192 generated tokens

These exploratory runs use the original ten numerical bases, the pinned
OLMo-3-7B-Think revision, bf16, temperature 0.6, top-p 0.95, an 8,192-token
generation limit, and a 16,384-token context. They are separate from the
earlier 4,096-token Stage 1 grid.

All six Stage 1 responses that previously hit the token limit were rerun.
All six now finished with `stop`: five correctly recognized underdetermination
(four symbolic expressions and one verbal abstention), while the omitted-start
prompt for base 1 gave the predicted unsupported numerical answer of 1. One
additional one-prompt GPU-fit smoke run also completed correctly; it duplicates
the first item of the six-prompt rerun and is not counted as an independent
base. The two runs gave different valid final forms on that item, illustrating
sampling variation even with the same nominal seed in different batches.

The new marked-unknown baseline says the agent has a known number of objects
in a pocket **and some more in a closed box**, then receives and gives away
known amounts, and asks for the total now. If the pocket amount is `p`, the
received amount `r`, the given-away amount `g`, and the box contains an
unspecified `x`, the correct result is `p + r − g + x`; a unique number is not
determined. The ten-base run produced seven correct abstentions and **three
completed unsupported numbers**. Each incorrect number was `p + r − g`, as if
the box's contents contributed zero. In those three responses the model treated
the box as irrelevant to the total, despite the question asking for the total
number of objects.

The same ten pocket-plus-box prompts were then run with the assumption
instruction in the system message and appended to the user message. Both runs
used the same model revision and 8,192/16,384-token limits as the baseline.
All 30 responses across the three placements completed with `stop`.

| Placement | Correctly underdetermined | Unsupported pocket-only numbers |
| --- | ---: | ---: |
| No added instruction | 7/10 | 3/10 |
| System instruction | 9/10 | 1/10 |
| User instruction | 10/10 | 0/10 |

The system instruction corrected baseline errors on bases 3 and 9; base 7
still excluded the box. User placement corrected all three baseline errors.
This supplies the missing **exploratory instruction positive control** for
OLMo: a marked missing quantity caused completed errors at baseline, and the
instruction reduced those errors without truncations. It does not by itself
show that the same instruction reliably fixes omissions, or establish a stable
effect size from one sampled response per cell. A later confirmatory test must
use fresh bases.

The local diagnostic scorer v5 regraded raw responses without changing the
saved JSONL files. One user-placement response ended with the standalone line
“cannot be determined” but omitted the requested `Final answer:` label; v5
correctly counts it as an abstention. Without that narrow correction, the
user-placement score in the saved file is 9/10.
