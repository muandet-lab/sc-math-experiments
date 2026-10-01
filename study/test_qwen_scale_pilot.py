import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from study.run_gpu_pilot import MODELS, run as run_gpu_pilot, sampling_settings
from study.run_qwen_scale_pilot import QWEN_SIZES, run


class QwenScalePilotTests(unittest.TestCase):
    def test_five_sizes_use_thinking_and_separate_outputs(self):
        self.assertEqual(tuple(QWEN_SIZES), ("1.7b", "4b", "8b", "14b", "32b"))
        with tempfile.TemporaryDirectory() as directory:
            with patch("study.run_qwen_scale_pilot.run_gpu_pilot") as pilot:
                outputs = [run(size, Path(directory)) for size in QWEN_SIZES]
        self.assertEqual(len(set(outputs)), 5)
        for (size, model_key), output in zip(QWEN_SIZES.items(), outputs):
            self.assertEqual(MODELS[model_key], f"Qwen/Qwen3-{size.upper()}")
            self.assertEqual(output.name, f"qwen3-{size}-bf16-thinking-pilot.jsonl")
        self.assertEqual([call.args[1] for call in pilot.call_args_list],
                         ["thinking"] * 5)
        self.assertTrue(all(sampling_settings(key, "thinking")["temperature"] == 0.6
                            for key in QWEN_SIZES.values()))

    def test_new_sizes_reject_non_thinking_before_gpu_loading(self):
        with self.assertRaisesRegex(ValueError, "require thinking"):
            run_gpu_pilot("qwen4b", "non-thinking", Path("unused.jsonl"))


if __name__ == "__main__":
    unittest.main()
