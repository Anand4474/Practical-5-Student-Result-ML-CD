import json
import unittest
from pathlib import Path
import joblib
import pandas as pd

class TestMLPipeline(unittest.TestCase):
    def test_model_metrics_and_dataset_exist(self):
        self.assertTrue(Path("student_result_model.pkl").exists())
        self.assertTrue(Path("metrics.json").exists())
        self.assertTrue(Path("student_results.csv").exists())

    def test_metrics_are_valid(self):
        with open("metrics.json", encoding="utf-8") as f:
            metrics = json.load(f)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["training_records"], 0)
        self.assertGreater(metrics["testing_records"], 0)

    def test_model_can_predict(self):
        model = joblib.load("student_result_model.pkl")
        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80,
        }])
        prediction = int(model.predict(sample)[0])
        self.assertIn(prediction, [0, 1])

if __name__ == "__main__":
    unittest.main()
