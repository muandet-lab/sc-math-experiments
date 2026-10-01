# Case-5 behavioral diagnostic pilot: Qwen3-0.6B and OLMo-3-7B-Think

The local files contain ten fresh bases and nine independent prompts per base
for each model. Qwen3-0.6B used thinking mode; OLMo-3-7B-Think used native
thinking. Both used bf16, 4,096 maximum generated tokens, 8,192 maximum
context tokens, temperature 0.6, and top-p 0.95. Qwen additionally used
top-k 20. All 90 Qwen generations finished with `stop`; OLMo had 89 `stop`
responses and one `length` truncation.

| Variant | Qwen correct | OLMo correct | Qwen / OLMo unsupported numbers |
| --- | ---: | ---: | ---: |
| Complete solve | 10/10 | 10/10 | 0 / 0 |
| Initial amount omitted | 0/10 | 0/10 | 10 / 10 |
| Received amount unspecified | 7/10 | 9/10 | 3 / 0 |
| Given-away amount unspecified | 5/10 | 10/10 | 5 / 0 |
| Initial amount explicitly unspecified | 10/10 | 10/10 | 0 / 0 |
| Initial amount explicitly zero | 10/10 | 10/10 | 0 / 0 |
| Ask for net change with unknown initial amount | 10/10 | 10/10 | 0 / 0 |
| Complete problem, ask answerability | 10/10 | 10/10 | 0 / 0 |
| Omitted initial amount, ask answerability | 0/10 | 6/10 | 0 / 0 |

On every omitted-initial solve prompt, both models returned `received − given`,
the number obtained by treating the missing starting amount as zero. For
example, `case5diag-00000-missing_initial_solve` says Eli receives 5 shells and
gives away 2, without stating his starting count; both models answered 3.
This repeated pattern is consistent with an initial-state default, but the
output alone cannot establish which internal assumption produced it.

Explicitly writing “starts with an unspecified number” changed the result:
both models correctly recognized all ten as underdetermined. Qwen used a
correct expression in `x` on five and an abstention on five; OLMo used a
correct expression on all ten. Both solved all ten zero-start and net-change
controls. This argues against a general inability to compute the arithmetic
or a blanket tendency to reject any prompt containing “unspecified.” The
missing-received and missing-given variants still produced some unsupported
numbers for Qwen, so the phenomenon is not exclusive to initial-state
omission. Their “some” wording differs from the omitted opening fact, which
limits a pure location comparison.

The answerability intervention did not rescue Qwen's omitted-initial prompts:
it answered **yes** on all ten, while answering yes correctly on all ten
complete prompts. OLMo answered **no** correctly on six of ten omitted-initial
prompts even though it guessed a number when solving those same six bases.
This is evidence for a task-cue-dependent gap between detecting insufficiency
and using that judgment when producing an answer for OLMo, but not for Qwen in
this pilot. It does not establish an internal mechanism.

One original Qwen answerability row was marked unparseable because its final
answer began “yes” but later used “No other scenario” in its explanation.
Scorer v2 fixed that. Scorer v3 additionally handles OLMo's final answers
formatted as `** x + 1` or `\( x + 2 \)`, correcting two false unsupported
number labels. The saved JSONL files remain untouched. All figures above are
recomputed from raw responses with scorer v3. OLMo's missing-received item for
base 6 hit the token limit before a final answer; the other 89 completed
responses were parseable.

The result is a 10-base exploratory pilot with one sampled response per
variant. Fresh held-out numerical instances and wording variations are needed
before generalizing beyond these templates.

```sh
python3 -m study.summarize_case5_diagnostics \
  study/outputs/qwen-case5-diagnostics-pilot.jsonl \
  study/olmo-case5-diagnostics-pilot.jsonl
```
