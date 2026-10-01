# Exploratory bf16 A100 pilot

These are the 20 matched renderings of the five base problems from the local
pilot (IDs 2, 3, 5, 6, and 7), not the planned full benchmark. Each base has
verbal direct, verbal reversed, symbolic direct, and symbolic reversed forms.
All four forms have the same final answer. The pilot seed, `20260929`, and
these items are excluded from final evaluation.

Runs used one A100, bf16 weights, vLLM 0.30.0, PyTorch 2.13.0+cu130, and a
2,048-token generation cap. The Qwen3-0.6B revision was
`c1899de289a04d12100db370d81485cdf75e47ca`; the OLMo 3 7B Think revision
was `d97e442d7cc678210054dbcc9b440894d62c89a4`. Qwen non-thinking used
temperature 0.7, top-p 0.8, top-k 20. Qwen thinking used temperature 0.6,
top-p 0.95, top-k 20. OLMo native thinking used temperature 0.6 and top-p
0.95. All runs used sampling seed `20260929`. Thus the Qwen mode comparison
also changes sampling settings, and the model comparison changes model size
and training; neither isolates those factors.

| Configuration | Verbal direct | Verbal reversed | Symbolic direct | Symbolic reversed | Total | Verbal gap | Symbolic gap | Difference of gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B non-thinking | 3/5 | 2/5 | 2/5 | 1/5 | 8/20 | +20 pp | +20 pp | 0 pp |
| Qwen3-0.6B thinking | 5/5 | 5/5 | 5/5 | 4/5 | 19/20 | 0 pp | +20 pp | −20 pp |
| OLMo 3 7B Think native thinking | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 | 0 pp | 0 pp | 0 pp |

The verbal gap is direct minus reversed accuracy; the symbolic gap is the
same difference without relational keywords. The difference of gaps is the
study's descriptive keyword-specific estimate, Δ_kw. Each cell has only five
items, so a single answer changes a gap by 20 percentage points. No output
was unparseable or stopped at the token limit; all 60 had finish reason
`stop`.

The one Qwen thinking error is `00002-symbolic-rev`: it answered 12 instead
of 7. Its trace correctly finds Maya has 6 + 2 = 8 and solves
`8 = Theo − 4` to get Theo = 12. It then gives 12 as the final answer, omitting
the subsequent sentence, “Theo gives away 5 coins.” The matched verbal
reversed trace performs that final subtraction and answers 7. This error is
therefore a post-comparison step omission, not evidence of following a
relational keyword.

Qwen non-thinking has errors across all four cells. Its wrong
`00002-verbal-rev` answer of 10 also omits steps: it computes Theo = 6 + 4
but skips both Maya receiving 2 and Theo giving away 5. Several `00005`
answers of 16 omit Eli receiving 3. Its `00006-symbolic-rev` answer is 1
instead of 13 even though the symbolic form has no keyword. These examples
show that final-answer accuracy alone cannot identify which step failed.
Its symbolic accuracy is 3/10, below the action plan's
50% floor threshold, so it needs a larger pilot and step-level
inspection before using its difference of gaps to support a shortcut claim.

The thinking runs are at or near ceiling on these short items. The data do
not establish either presence or absence of a keyword shortcut in a broader
problem distribution. The action plan's harder multi-comparison items,
larger balanced pilot, and intermediate-step labels remain necessary.
Sampling and precision also differ from the earlier 4-bit MLX pilot, so the
change in Qwen results cannot be attributed to bf16 alone.

Recompute this table and list the wrong item IDs with
`python3 -m study.summarize_gpu_pilot` from the repository root. Complete
responses are in the three ignored JSONL files under `study/outputs/`.
