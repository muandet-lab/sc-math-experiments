# Matched diagnostics for shortcut cases 2–6

This draft records the five follow-up diagnostics beyond case 1, additive
keyword reversal. The implementation is in `generate_shortcut_cases.py`; the
thinking-only vLLM runner is `run_shortcut_cases.py`. This is exploratory
design work and is not a frozen pre-registration. Use a fresh private seed
for any later evaluation set, then freeze the generator revision, sample
sizes, models, prompts, decoding settings, and analysis before running it.
The earlier `study.generate` pilot mixed additive and multiplicative
comparisons, so its aggregate results are not an additive-only case-1
baseline. Case 2 here isolates multiplication and division.

Each case is generated as a pair with one intended manipulation and an
independently stored gold response. `--count N` yields `2N` prompts. Agent
names, entities, amounts, and arithmetic operations vary within a case.
All intermediate values in solvable items are bounded between 2 and 20.

| Case | Pair and intended manipulation | Primary descriptive measurement |
| --- | --- | --- |
| 2. Multiplicative relation reversal | Same problem and gold answer; comparison changes from direct “times as many”/reciprocal wording to its logically equivalent reversed form. Additive reversals are excluded. | Direct minus reversed accuracy; inspect wrong comparison values for surface-operation signatures. |
| 3. Comparison versus transfer | Same queried agent, amount, operation, and answer. One form changes that agent's count by a transfer; the other infers it from a static comparison with another agent. | Transfer minus comparison accuracy. |
| 4. Sentence-order bias | Same starting fact, two static comparison facts, question, and answer. Only the order of the two comparison sentences changes. | Dependency-order minus reverse-order accuracy. |
| 5. Solvability prior | Remove only the opening numerical premise from a solvable arithmetic problem; the remaining statements contain no numerical anchor for the unknown starting value. | Correct “cannot be determined” rate and numerical-guess rate on the missing-premise form. |
| 6. Familiar solution-template bias | Keep all facts fixed and change only the final question from the target agent to the source agent. Both answers are integers and are required to differ. There is a transfer before and after the comparison, so neither query is a direct lookup. | Changed-query accuracy and the rate at which it repeats the original answer. |

Case 3 cannot hold every surface feature fixed: the transfer form states the
queried agent's starting count, whereas the static form states the source
agent's count. Its gap is a task-format diagnostic, not an isolated estimate
of a keyword effect. Case 5 changes answer type as part of the solvability
manipulation. Case 6's “repeated original answer” is a useful signature,
but the model's trace must still be inspected before calling an error a
template shortcut. The sentence-order pair uses simultaneous “has” facts,
so reversing their textual order cannot change the mathematical answer.

Both model choices use reasoning: Qwen3-0.6B receives the explicit
`enable_thinking=True` chat-template setting; OLMo 3 7B Think uses its native
thinking template. Raw responses, prompts, gold answers, model revisions,
and decoding settings are saved in JSONL. The parser accepts a final integer
or a final “cannot be determined” response and counts missing final answers
separately from incorrect answers. Truncated and unparseable outputs should
be reported by case and variant.

The generator's structural checks and unit tests verify paired facts and
answers. Human review is still needed to confirm naturalness, particularly
for reciprocal wording and missing-premise items. A larger pilot must check
floor and ceiling accuracy, parser behavior, and output-token adequacy before
these diagnostics support inferential claims.
