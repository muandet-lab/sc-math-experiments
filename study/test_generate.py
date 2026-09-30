import random
import unittest
from copy import deepcopy
from collections import Counter

from study.generate import OPERATIONS, generate, generate_base, validate_item


class GeneratorTests(unittest.TestCase):
    def test_four_way_pairing_and_ground_truth(self):
        rows = generate(80, 8361)
        self.assertEqual(len(rows), 320)
        self.assertEqual(Counter(row["operation"] for row in rows),
                         {operation: 80 for operation in OPERATIONS})
        for offset in range(0, len(rows), 4):
            group = rows[offset:offset + 4]
            self.assertEqual({(row["form"], row["reversal"]) for row in group},
                             {(form, reversal) for form in ("verbal", "symbolic")
                              for reversal in (False, True)})
            self.assertEqual(len({row["answer"] for row in group}), 1)
            self.assertEqual(len({row["comparison_step"] for row in group}), 1)
            for row in group:
                validate_item(row)

    def test_each_operation_and_step_position(self):
        for operation in OPERATIONS:
            positions = set()
            for base_id in range(120):
                group = generate_base(random.Random(base_id), base_id, operation)
                positions.add(group[0]["comparison_step"])
                self.assertTrue(all(2 <= row["answer"] <= 20 for row in group))
            self.assertGreaterEqual(len(positions), 4)

    def test_reproducibility(self):
        self.assertEqual(generate(8, 123), generate(8, 123))
        self.assertNotEqual(generate(8, 123), generate(8, 124))

    def test_integrity_rejects_rewritten_relation(self):
        row = deepcopy(generate(1, 876)[1])
        row["sentences"][row["comparison_step"]] = "Nora has 5 more tokens than Eli."
        row["problem"] = " ".join(row["sentences"])
        with self.assertRaises(ValueError):
            validate_item(row)


if __name__ == "__main__":
    unittest.main()
