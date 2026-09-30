import unittest

from study.run_local_pilot import extract_answer


class ExtractionTests(unittest.TestCase):
    def test_final_answer_formats(self):
        self.assertEqual(extract_answer("</think>\nFinal answer: 7"), 7)
        self.assertEqual(extract_answer("**Final Answer:**\n<7>"), 7)
        self.assertEqual(extract_answer("Answer: $\\boxed{7}$"), 7)
        self.assertEqual(extract_answer("**Answer:** 4 marbles."), 4)

    def test_no_final_is_unparsed(self):
        self.assertIsNone(extract_answer("12 - 5 = 7. Let me check."))
        self.assertIsNone(extract_answer("Final answer: <15.25>"))
        self.assertIsNone(extract_answer("Final answer: 0.5"))


if __name__ == "__main__":
    unittest.main()
