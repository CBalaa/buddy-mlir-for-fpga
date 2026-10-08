"""LLM graph contract tests are not run for EfficientNet vision."""
import unittest


class VisionGraphContracts(unittest.TestCase):
    def test_llm_contract_suite_not_applicable(self):
        self.skipTest(
            "Qwen graph_contract / attention tests do not apply; "
            "see triton_call_replace.py and test_quant_reference.py"
        )


if __name__ == "__main__":
    unittest.main()
