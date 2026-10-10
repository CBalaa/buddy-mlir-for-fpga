"""Vision image ABI: default FP32 uses build_model_image.py; LLM prefill/decode N/A."""
import unittest


class VisionImageABI(unittest.TestCase):
    def test_w8a8_llm_launch_abi_not_applicable(self):
        self.skipTest(
            "EfficientNet FP32 uses build_model_image.py; "
            "W8A8 board launch for vision is not LLM-shaped"
        )


if __name__ == "__main__":
    unittest.main()
