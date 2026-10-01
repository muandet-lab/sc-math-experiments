# Case-5 assumption-instruction follow-up: Qwen and OLMo

The local intervention files contain 30 single-turn responses per model: ten
matched bases in each of three existing conditions. They use the same model
revisions, questions, bf16 precision, sampling settings, 4,096-token generation
limit, and 8,192-token context limit as the original diagnostic pilot. Every
system prompt additionally says:

> Use only the stated facts. Do not assign values to quantities whose values
> are not given. If the requested quantity is not uniquely determined, state
> that.

| Condition | Qwen baseline → intervention | OLMo baseline → intervention |
| --- | ---: | ---: |
| Complete solve | 10/10 → 10/10 | 10/10 → 10/10 |
| Initial amount omitted | 0/10 → 0/10 | 0/10 → 1/10 |
| Net change with unknown initial amount | 10/10 → 10/10 | 10/10 → 10/10 |

All 60 intervention responses finished with `stop`; none were truncated. On
every omitted-start problem Qwen still gave `received − given`, the unsupported
number equal to the result if the missing starting amount were zero. For
example, when Eli receives 5 shells and gives away 2, Qwen again answered 3.
OLMo did the same on nine of ten bases. Its one corrected example was Eli: the
baseline answer was 3, while the intervention answer was “cannot be
determined.” Thus the instruction produced one selective correction for OLMo
and none for Qwen, without over-abstention on the two controls. The omitted-
start failure persisted on 19 of 20 sampled intervention responses across
models.

The saved OLMo file initially labeled one net-change control as an error
because the correct `\boxed{1}` appeared on the line after `Final answer:`.
Scorer v4 recognizes this formatting and restores that control to 10/10. The
raw JSONL files remain untouched; results above are regraded from raw
responses.

One response per matched prompt is exploratory. The results show persistence
despite an explicit instruction, but do not establish whether the behavior
reflects a pragmatic default, a shortcut, or another internal mechanism.

```sh
python3 -m study.summarize_case5_assumption_pilot \
  --baseline study/outputs/qwen-case5-diagnostics-pilot.jsonl \
  --intervention study/qwen-case5-assumption-pilot.jsonl
python3 -m study.summarize_case5_assumption_pilot \
  --baseline study/olmo-case5-diagnostics-pilot.jsonl \
  --intervention study/olmo-case5-assumption-pilot.jsonl
```
