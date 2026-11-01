"""
Unit tests for SLM Fine-Tuning and Evaluation Suite (SLM-1).
Tests LoRA configuration, training pipeline verification, inference modes, and comparative benchmarking.
"""

import unittest
import os
import shutil
import tempfile

from src.models.lora_config import SLMTrainingConfig
from src.models.train_slm import SLMTrainer
from src.models.inference import SLMInferenceEngine
from src.models.benchmark import LegalModelBenchmark
from src.data.pipeline import DataPipeline
from src.data.schema import ChatTurn


SAMPLE_EVAL_CONTRACT = """MASTER SERVICES AND INTELLECTUAL PROPERTY AGREEMENT

This Agreement is made on April 1, 2024 between Apex Systems Inc ("Buyer") and Zenith Tech Ltd ("Provider").

1. TERMINATION
Either party may terminate this Agreement without cause upon sixty (60) days prior written notice. Immediate termination is permitted upon insolvency or uncured material breach.

2. LIABILITY CAP
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR INDIRECT OR CONSEQUENTIAL DAMAGES. AGGREGATE LIABILITY IS CAPPED AT TOTAL FEES PAID IN THE PRIOR 12 MONTHS.

3. INDEMNITY
Provider will defend and indemnify Buyer against third-party patent and trademark infringement claims.

4. GOVERNING LAW
This Agreement is governed by the laws of England and Wales, with LCIA arbitration in London.
"""


class TestSLMAndEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.sft_dir = os.path.join(self.temp_dir, "sft")
        self.reports_dir = os.path.join(self.temp_dir, "reports")
        # Generate temporary SFT data
        pipeline = DataPipeline(raw_base_dir=os.path.join(self.temp_dir, "raw"), sft_output_dir=self.sft_dir)
        pipeline.run()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_lora_config_and_peft_params(self):
        """FR-9: Verify LoRA parameters (rank, alpha, target modules, causal LM task)."""
        config = SLMTrainingConfig(
            base_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            lora_r=16,
            lora_alpha=32,
            output_dir=os.path.join(self.temp_dir, "checkpoints"),
        )
        self.assertEqual(config.lora_r, 16)
        self.assertEqual(config.lora_alpha, 32)
        self.assertIn("q_proj", config.target_modules)
        self.assertIn("v_proj", config.target_modules)

        peft_cfg = config.get_peft_config()
        self.assertIsNotNone(peft_cfg)

    def test_slm_training_pipeline_verification(self):
        """FR-9, AC-8: Verify SFT dataset validation and training orchestration."""
        train_path = os.path.join(self.sft_dir, "train.jsonl")
        trainer = SLMTrainer(
            config=SLMTrainingConfig(output_dir=os.path.join(self.temp_dir, "checkpoints"))
        )
        count = trainer.validate_dataset_ready(train_path)
        self.assertGreater(count, 0)

        run_res = trainer.run_training(train_path=train_path, dry_run=True)
        self.assertEqual(run_res["status"], "success")
        self.assertEqual(run_res["training_examples"], count)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "checkpoints", "training_manifest.json")))

    def test_inference_engine_modes(self):
        """FR-9: Test baseline vs fine-tuned response generation."""
        engine_base = SLMInferenceEngine(is_finetuned=False)
        engine_slm = SLMInferenceEngine(is_finetuned=True)

        messages = [ChatTurn(role="user", content="What is the liability cap?")]
        evidence = "Aggregate liability is capped at total fees paid in the prior 12 months."

        resp_base = engine_base.generate_response(messages, context_evidence=evidence)
        resp_slm = engine_slm.generate_response(messages, context_evidence=evidence)

        self.assertIn("capped at total fees", resp_base)
        self.assertIn("CGL Contract Analysis", resp_slm)
        self.assertIn("Specialized SLM", resp_slm)

    def test_comparative_benchmark(self):
        """AC-8: Benchmark pretrained baseline vs fine-tuned SLM on identical contract test data."""
        bench = LegalModelBenchmark(output_dir=self.reports_dir)
        report = bench.run_benchmark(SAMPLE_EVAL_CONTRACT)

        self.assertIn("baseline", report)
        self.assertIn("finetuned_slm", report)
        self.assertIn("relative_improvement", report)

        baseline = report["baseline"]
        slm = report["finetuned_slm"]

        # Fine-tuned SLM must show measurable improvement over baseline
        self.assertGreater(slm["clause_extraction_f1"], baseline["clause_extraction_f1"])
        self.assertGreater(slm["nli_accuracy"], baseline["nli_accuracy"])
        self.assertGreater(slm["risk_precision"], baseline["risk_precision"])
        self.assertGreater(slm["faithfulness_rate"], baseline["faithfulness_rate"])

        # Faithfulness rate must be high (low hallucination)
        self.assertGreaterEqual(slm["faithfulness_rate"], 0.95)

        self.assertTrue(os.path.exists(os.path.join(self.reports_dir, "benchmark_report.json")))


if __name__ == "__main__":
    unittest.main()
