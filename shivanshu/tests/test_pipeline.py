#!/usr/bin/env python3
"""
Automated Integration & Regression Test Suite for Polymer Classification.
Tests:
1. Class taxonomy and label consistency
2. PyTorch model loading and inference (ConvNeXt-Tiny)
3. Native pure NumPy engine inference (s390x IBM LinuxONE)
4. Composition estimator outputs (predictions.json, confusion_matrix.json)
5. GitHub file size limits (< 100 MB per file)
"""

import os
import sys
import glob
import json
import unittest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPolymerPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_images = sorted(glob.glob(os.path.join(BASE_DIR, "dataset_crops", "test", "*", "*.jpg")))
        if not cls.test_images:
            cls.test_images = sorted(glob.glob(os.path.join(BASE_DIR, "assets", "sample_images", "*.jpg")))
        assert len(cls.test_images) > 0, "No test images found in dataset_crops/test or assets/sample_images!"
        cls.sample_image = cls.test_images[0]

    def test_01_class_taxonomy(self):
        """Verifies canonical 6-class mapping and JSON config."""
        mapping_path = os.path.join(BASE_DIR, "class_mapping.json")
        self.assertTrue(os.path.exists(mapping_path))
        with open(mapping_path) as f:
            data = json.load(f)
        self.assertEqual(data["idx_to_class"]["0"], "PE-HD")
        self.assertEqual(data["idx_to_class"]["2"], "PET")
        self.assertEqual(len(data["idx_to_class"]), 6)

    def test_02_pytorch_inference(self):
        """Verifies PyTorch ConvNeXt-Tiny inference and probability outputs."""
        try:
            import torch
            from polymer_classifier.infer import PolymerPredictor

            ckpt_path = os.path.join(BASE_DIR, "models", "best_polymer_model.pth")
            self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")

            predictor = PolymerPredictor.from_checkpoint(ckpt_path)
            res = predictor.predict(self.sample_image, top_k=3)

            self.assertIn("predicted_class", res)
            self.assertIn("confidence", res)
            self.assertIn("all_probabilities", res)
            self.assertEqual(len(res["all_probabilities"]), 6)

            prob_sum = sum(res["all_probabilities"].values())
            self.assertAlmostEqual(prob_sum, 1.0, places=3)
            print(f"  [PyTorch] Test image classified as {res['predicted_class']} ({res['confidence']*100:.1f}%)")
        except ImportError:
            self.skipTest("PyTorch not installed in this environment (expected on s390x)")

    def test_03_numpy_engine_inference(self):
        """Verifies pure NumPy engine inference without PyTorch/CUDA."""
        from polymer_classifier.infer import NumPyPolymerPredictor

        npz_path = os.path.join(BASE_DIR, "models", "model_weights.npz")
        self.assertTrue(os.path.exists(npz_path), f"NumPy weights missing: {npz_path}")

        predictor = NumPyPolymerPredictor(npz_path)
        res = predictor.predict(self.sample_image, top_k=3)

        self.assertIn("predicted_class", res)
        self.assertIn("confidence", res)
        self.assertIn("all_probabilities", res)
        self.assertEqual(len(res["all_probabilities"]), 6)

        prob_sum = sum(res["all_probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)
        print(f"  [NumPy] Test image classified as {res['predicted_class']} ({res['confidence']*100:.1f}%)")

    def test_04_composition_estimator_files(self):
        """Verifies data/predictions.json and data/confusion_matrix.json schemas."""
        pred_path = os.path.join(BASE_DIR, "data", "predictions.json")
        cm_path = os.path.join(BASE_DIR, "data", "confusion_matrix.json")

        self.assertTrue(os.path.exists(pred_path), f"Missing: {pred_path}")
        self.assertTrue(os.path.exists(cm_path), f"Missing: {cm_path}")

        # Check predictions
        with open(pred_path) as f:
            preds = json.load(f)
        self.assertIsInstance(preds, list)
        self.assertGreater(len(preds), 0)

        item = preds[0]
        self.assertIn("image", item)
        self.assertIn("material", item)
        self.assertIn("polymer", item)
        self.assertIn("confidence", item)
        self.assertIn("probs", item)
        self.assertAlmostEqual(sum(item["probs"].values()), 1.0, places=3)

        # Check confusion matrix
        with open(cm_path) as f:
            cm = json.load(f)
        self.assertIn("source", cm)
        self.assertNotIn("SYNTHETIC", cm["source"])
        self.assertEqual(cm["labels"], ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"])
        self.assertEqual(len(cm["matrix"]), 7)
        for row in cm["matrix"]:
            self.assertEqual(len(row), 7)

    def test_05_github_size_compliance(self):
        """Verifies no single tracked model file exceeds GitHub's 100 MB limit."""
        ckpt_path = os.path.join(BASE_DIR, "models", "best_polymer_model.pth")
        if os.path.exists(ckpt_path):
            size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            self.assertLess(size_mb, 100.0, f"Checkpoint {size_mb:.2f} MB exceeds GitHub 100MB limit!")
            print(f"  [Git Check] Model checkpoint is {size_mb:.2f} MB (< 100 MB limit)")


if __name__ == "__main__":
    unittest.main()
