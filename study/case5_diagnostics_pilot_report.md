# Case-5 behavioral diagnostic pilot: Qwen3-0.6B

The local files contain ten fresh bases and nine independent prompts per base
for Qwen3-0.6B in thinking mode. The saved configuration is bf16, 4,096 maximum
generated tokens, 8,192 maximum context tokens, temperature 0.6, top-p 0.95,
and top-k 20. All 90 generations finished with `stop`; none hit the token cap.
The OLMo diagnostic response file is not present locally, so this report
covers Qwen only.

| Variant | Correct | Completed unsupported numbers | Zero-equivalent numbers |
| --- | ---: | ---: | ---: |
| Complete solve | 10/10 | 0 | 0 |
| Initial amount omitted | 0/10 | 10 | 10 |
| Received amount unspecified | 7/10 | 3 | 0 |
| Given-away amount unspecified | 5/10 | 5 | 5 |
| Initial amount explicitly unspecified | 10/10 | 0 | 0 |
| Initial amount explicitly zero | 10/10 | 0 | 0 |
| Ask for net change with unknown initial amount | 10/10 | 0 | 0 |
| Complete problem, ask answerability | 10/10 | 0 | 0 |
| Omitted initial amount, ask answerability | 0/10 | 0 | 0 |

On every omitted-initial solve prompt, Qwen returned `received − given`, the
number obtained by treating the missing starting amount as zero. For example,
`case5diag-00000-missing_initial_solve` says Eli receives 5 shells and gives
away 2, without stating his starting count; Qwen answered 3. This repeated
pattern is consistent with an initial-state default, but the output alone
cannot establish which internal assumption produced it.

Explicitly writing “starts with an unspecified number” changed the result:
Qwen correctly recognized all ten as underdetermined, using a correct
expression in `x` on five and an abstention on five. It also solved all ten
zero-start and net-change controls. This argues against a general inability to
compute the arithmetic or a blanket tendency to reject any prompt containing
“unspecified.” The missing-received and missing-given variants still produced
some unsupported numbers, so the phenomenon is not exclusive to initial
state omission. Their “some” wording differs from the omitted opening fact,
which limits a pure location comparison.

The answerability intervention did not rescue omitted-initial prompts: Qwen
answered **yes** on all ten, while answering yes correctly on all ten complete
prompts. Thus this pilot does not show a gap in which it detects missing
information when asked directly but ignores that judgment while solving.
It more strongly suggests that the omitted initial quantity was not represented
as a necessary unknown under either question wording. That is a behavioral
interpretation, not an internal-mechanism claim.

One original answerability row was marked unparseable because its final answer
began “yes” but later used “No other scenario” in its explanation. Scorer v2
reads the leading yes/no on the final line and regrades that row correctly;
the saved JSONL remains untouched. All figures above are recomputed from raw
responses with scorer v2.

The result is a 10-base exploratory pilot with one sampled response per
variant. A fresh held-out numerical set, the OLMo diagnostic run, and manual
review of unsupported answers are needed before generalizing beyond these
templates.

```sh
python3 -m study.summarize_case5_diagnostics \
  study/outputs/qwen-case5-diagnostics-pilot.jsonl
```
