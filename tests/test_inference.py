"""Run with: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_utils import (
    PatientProfile, encode_profile, grouped_importances,
    load_artifacts, predict_profile,
)


class ModelCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.scaler, cls.columns = load_artifacts()

    def test_feature_order(self):
        encoded = encode_profile(PatientProfile(), self.columns)
        self.assertEqual(encoded.shape, (1, 15))
        self.assertEqual(list(encoded.columns), self.columns)
        self.assertEqual(encoded.loc[0, "gender_Female"], 1)
        self.assertEqual(encoded.loc[0, "smoking_history_never"], 1)

    def test_prediction_and_score(self):
        outcome, score = predict_profile(
            PatientProfile(), self.model, self.scaler, self.columns
        )
        self.assertIn(outcome, (0, 1))
        self.assertTrue(score is None or (0 <= score <= 1))

    def test_importances_add_to_one(self):
        values = grouped_importances(self.model, self.columns)
        self.assertAlmostEqual(values["Importance"].sum(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
