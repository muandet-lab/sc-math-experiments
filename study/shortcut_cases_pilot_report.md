# Thinking-mode shortcut diagnostics: local pilot

These exploratory results use ten matched pairs per case and model. The saved
JSONL responses in `study/outputs/` contain the pinned model revisions, prompts,
raw generations, finish reasons, and original grades. The figures below were
recomputed from the raw generations with `study.summarize_shortcut_cases`; the
recomputation does not modify the saved files. Accuracy requires a parseable
final answer outside any completed thinking block. A generation that reaches
its token limit before closing `</think>` is ungraded as a parse failure.

| Case | Qwen3-0.6B control → challenge | OLMo-3-7B-Think control → challenge |
| --- | --- | --- |
| 2. Multiplicative reversal | 6/10 → 5/10 | 8/10 → 9/10 |
| 3. Transfer → static comparison | 10/10 → 10/10 | 10/10 → 10/10 |
| 4. Dependency → reversed sentence order | 10/10 → 10/10 | 10/10 → 10/10 |
| 5. Solvable → missing premise | 7/10 → 1/10 | 7/10 → 1/10 |
| 6. Original → changed query | 6/10 → 10/10 | 6/10 → 5/10 |

The original parser only recognized `Final answer:` at the start of a line.
OLMo often put it after explanatory text on the same line. Regrading changed
7 OLMo case-3 rows and 3 case-4 rows; both apparent effects disappeared. The
parser also now ignores answer-like text inside an unfinished thinking block.

Case 2 does not show a consistent reversal cost. OLMo's three errors are all
token-limit truncations. Qwen has two parse failures and one truncation in
each variant, leaving only a one-item accuracy gap. Cases 3 and 4 are at the
ceiling for both models; these items cannot measure whether the proposed
shortcuts appear on harder problems.

Case 5 contains completed, incorrect numerical answers to genuinely
underdetermined questions: 2/10 for Qwen and 3/10 for OLMo. For example,
`case5-00004-missing_premise` says Maya *receives* 5 coins and later gives
away 4, but never states how many she started with. Both models answered 4,
as if receiving 5 established her initial count. Both models correctly
abstained on only 1/10 missing-premise prompts. However, Qwen truncated 5/10
and OLMo truncated 6/10 missing-premise generations. Thus the large 7/10 to
1/10 accuracy gap combines solvability mistakes with a generation-limit
failure; it is not a clean estimate of the prior alone. The prompt's literal
`<integer>` placeholder also appears unfilled in several Qwen final answers.

Case 6 does not show the proposed familiar-template signature: neither model
repeated the original-query gold answer in a changed-query response. Qwen did
better on the changed query, while all five of OLMo's changed-query failures
were truncations. The two queries can have different intrinsic difficulty.

For a follow-up pilot, keep these files intact and use new output names. The
runner now asks for an actual whole number or the exact abstention phrase and
forbids placeholders; this is a **new prompt condition** and its results should
not be pooled with these files. The runner records a parse status for every
answer, and the summarizer separates completed responses without a parseable
final answer from token-limit truncations. Raise the response and context
limits together for cases 2, 5, and 6, then inspect completed wrong answers
separately from parsing and truncation. Cases 3 and 4 need harder matched
items before larger sample sizes help.

Reproduce the table and diagnostics locally with:

```sh
python3 -m study.summarize_shortcut_cases study/outputs/*shortcut-case*-thinking.jsonl
```
