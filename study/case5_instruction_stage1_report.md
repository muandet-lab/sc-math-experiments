# Case-5 incremental instruction-placement Stage 1

This exploratory run adds 13 previously unobserved cells on each of the ten
original case-5 bases: 130 responses per model. The saved original diagnostic
and assumption-instruction pilots fill the other cells. Model revisions,
sampling settings, and the 4,096-token generation limit match those earlier
pilots. `None` means no added instruction; `system` and `user` place the same
instruction in different chat roles. All figures below are regraded from raw
responses with diagnostic scorer v5.

| Condition | Qwen none / system / user | OLMo none / system / user |
| --- | --- | --- |
| Starting amount omitted | 0/10 · 0/10 · 0/10 | 0/10 · 1/10 · 3/10 |
| Box mentioned, no count | 6/10 · 9/10 · 6/10 | 7/10 · 10/10 · 10/10 |
| Tom's amount unresolved | 7/10 · 9/10 · 10/10 | 9/10 · 10/10 · 9/10 |
| “Gives away some” | 5/10 · 8/10 · 5/10 | 10/10 · 10/10 · 10/10 |
| Complete problem | 10/10 · 10/10 · 10/10 | 10/10 · 10/10 · 10/10 |
| Explicit zero start | 10/10 · 10/10 · 10/10 | 10/10 · 10/10 · 10/10 |
| Net change with unknown start | 10/10 · 10/10 · 10/10 | 10/10 · 10/10 · 10/10 |

Qwen completed all 130 Stage 1 responses. It gave the predicted unsupported
`received − given` answer on all ten omitted-start prompts under user-placed
instruction, just as it did in the two earlier conditions. Its system-placed
instruction improved the box condition from 6/10 to 9/10 and the vague
given-away condition from 5/10 to 8/10. User placement improved the
Tom-reference condition from 7/10 to 10/10, but did not improve box or vague
given-away prompts. No control showed over-abstention. The Qwen pattern is
consistent with instruction efficacy on some marked missing quantities and
resistance in the fully omitted-start condition, subject to the noise of one
sample per cell.

OLMo completed 124 of 130 Stage 1 responses; six hit the token limit. Its
user-placed instruction yielded three correct omitted-start answers, six
predicted numerical guesses, and one truncation. The system-placed instruction
had yielded one correct answer and nine guesses. OLMo's baseline box and Tom
shortfalls were entirely truncations (three and one, respectively); all
completed marked-condition responses were correct. It therefore lacks a
baseline **completed error** on these marked conditions that an instruction
could repair. Their apparent accuracy gains mainly reflect fewer truncations,
not demonstrated correction of a marked-unknown mistake. A harder marked
condition or a larger token budget would be needed for an OLMo positive
control.

One completed OLMo response correctly ended with “cannot be determined”
without the requested `Final answer:` label. Scorer v5 recognizes that narrow
case. The saved JSONL files were not edited. Six OLMo Stage 1 rows are
truncations; the remaining non-correct completed rows are unsupported numerical
answers. Qwen has no truncations. These pilot results do not establish a
training origin or, by themselves, justify calling the
behavior a spurious correlation.

```sh
python3 -m study.summarize_case5_instruction_stage1 \
  --baseline study/outputs/qwen-case5-diagnostics-pilot.jsonl \
  --system study/qwen-case5-assumption-pilot.jsonl \
  --stage1 study/qwen-case5-instruction-stage1.jsonl
python3 -m study.summarize_case5_instruction_stage1 \
  --baseline study/olmo-case5-diagnostics-pilot.jsonl \
  --system study/olmo-case5-assumption-pilot.jsonl \
  --stage1 study/olmo-case5-instruction-stage1.jsonl
```
