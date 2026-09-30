# Exploratory five-problem keyword-consistency pilot

Run date: 2026-09-29. This is an exploratory check of the action plan's
relation-keyword effect documented by [Opedal et al. (ICML 2024)](https://arxiv.org/abs/2401.18070), not a test of whether the underlying
training-data association exists. The latter still requires the planned
OLMo-corpus measurement. Each of five base problems has a bias-aligned
(`direct`) wording and a bias-conflicting (`reversal`) wording. A matched
symbolic pair removes the relational word while retaining the reversal.
All four renderings of a base problem have the same ground truth.

The generator used seed `20260929`, drew eight base problems, and selected
the five with at most three steps before this final comparison. Their IDs
are 2, 3, 5, 6, and 7. This selection reduces answer truncation in the
exploratory run and does not represent the planned 1–5-step distribution.
Responses were generated greedily with a 2,048-token cap for Qwen and a
1,024-token cap for OLMo. Both models used pinned 4-bit MLX conversions on a
local Apple GPU. Qwen3-0.6B used non-thinking mode; OLMo 3 7B Think used its
native thinking mode. Their different sizes, modes, and quantized weights
prevent a controlled cross-model scale comparison.
The conversions were [Qwen3-0.6B-4bit](https://huggingface.co/mlx-community/Qwen3-0.6B-4bit)
at revision `73e3e38d981303bc594367cd910ea6eb48349da8` and
[Olmo-3-7B-Think-4bit](https://huggingface.co/mlx-community/Olmo-3-7B-Think-4bit)
at revision `65294083d66d52f7bee42c5da768e3ad456332c4`.

| Model | Verbal direct | Verbal reversal | Symbolic direct | Symbolic reversal | Verbal gap | Symbolic gap | Difference of gaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | 4/5 | 1/5 | 2/5 | 2/5 | +60 pp | 0 pp | +60 pp |
| OLMo 3 7B Think | 5/5 | 5/5 | 5/5 | 5/5 | 0 pp | 0 pp | 0 pp |

Five verbal pairs, showing the changed comparison statement and the final
answer for the whole problem:

| ID | Bias-aligned statement | Bias-conflicting statement | Gold | Qwen aligned / conflicting | OLMo aligned / conflicting |
| --- | --- | --- | ---: | --- | --- |
| 2 | Theo has 4 more coins than Maya | Maya has 4 fewer coins than Theo | 7 | 7 / 3 | 7 / 7 |
| 3 | Amir has 1/4 as many cards as Theo | Theo has 4 times as many cards as Amir | 4 | no integer / 0 | 4 / 4 |
| 5 | Eli has 4 fewer marbles than Owen | Owen has 4 more marbles than Eli | 19 | 19 / 13 | 19 / 19 |
| 6 | Lena has 6 more cards than Nora | Nora has 6 fewer cards than Lena | 13 | 13 / 13 | 13 / 13 |
| 7 | Nora has 3 times as many coins as Iris | Iris has 1/3 as many coins as Nora | 12 | 12 / no answer | 12 / 12 |

The complete prompt includes starting quantities and any changes before or
after these statements; inspect `study.generate` or regenerate from the
recorded seed to see the full problems. For Qwen, four outputs did not provide
a parseable integer final answer and three hit the token cap. OLMo had no
parse failures or truncations. Neither model made a reversal error whose
final value equaled the predicted result from directly applying the displayed
keyword or operator to the known starting quantity (Qwen: 0/4 verbal and
0/3 symbolic reversal errors; OLMo: no reversal errors). Qwen's wrong answers
instead include omitting the comparison, omitting a later change, and
unfinished or repetitive reasoning. The +60-point difference of gaps is a
five-item descriptive value and **does not establish a causal keyword shortcut**.

This pilot shows the short items are at ceiling for OLMo and the Qwen setup
has output-format and reasoning failures. The next confirmatory run should
use the plan's larger balanced sample, validated parser and intermediate-step
labels, a sufficiently long generation limit, and bf16 weights where feasible.
The training-association claim also needs its separate corpus analysis.

Recompute the table with `python3 -m study.summarize_local_pilot`. Raw model
responses and model weights are kept under ignored `study/outputs/` and
`study/private/` directories respectively.
