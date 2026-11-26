"""Public-data-free checks of the disclosed TF-IDF / logistic-regression pipeline."""
import importlib.util
import pathlib
import tempfile
import unittest

import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_sentiment_model", ROOT / "scripts" / "train_sentiment_model.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class PipelineTests(unittest.TestCase):
    def test_tfidf_is_fitted_only_on_train_split(self):
        train = pd.Series(["clean comfortable room", "dirty noisy hotel", "friendly service"])
        test = pd.Series(["unseenwordonly clean room"])
        vectorizer, x_train, x_test = training.vectorize_text(train, test, max_features=100)
        self.assertEqual(x_train.shape[0], 3)
        self.assertEqual(x_test.shape[0], 1)
        self.assertNotIn("unseenwordonly", vectorizer.vocabulary_)

    def test_synthetic_binary_classification_and_external_scoring(self):
        train = pd.Series([
            "excellent clean comfortable", "friendly clean lovely",
            "great helpful peaceful", "wonderful good pleasant",
            "awful dirty noisy", "terrible rude dirty",
            "bad broken unpleasant", "poor noisy disappointing",
        ])
        labels = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
        vectorizer, train_vec, _ = training.vectorize_text(train, train.iloc[:1], 100)
        model = training.train_model(train_vec, labels)
        scored = pd.DataFrame({
            "Review": ["excellent clean", "awful dirty"],
            "Promotore": [1, 0],
        })
        old = training.preprocess_dataframe
        try:
            # Exercise the external-scoring API without requiring a network/NLTK download.
            training.preprocess_dataframe = lambda frame: frame["Review"]
            accuracy = training.evaluate_dataset(model, vectorizer, scored)
        finally:
            training.preprocess_dataframe = old
        self.assertEqual(accuracy, accuracy_score([1, 0], model.predict(vectorizer.transform(scored["Review"]))))
        self.assertGreaterEqual(accuracy, 0.5)

    def test_synthetic_fixture_contains_both_classes_and_no_identifying_records(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            # Keep fixture generation self-contained and independent of real review data.
            fixture_spec = importlib.util.spec_from_file_location("synthetic_fixture", ROOT / "scripts" / "generate_synthetic_fixture.py")
            fixture = importlib.util.module_from_spec(fixture_spec)
            fixture_spec.loader.exec_module(fixture)
            self.assertEqual(len(fixture.ROWS), 16)
            self.assertEqual({label for _, label in fixture.ROWS}, {0, 1})
            self.assertEqual(len({text for text, _ in fixture.ROWS}), len(fixture.ROWS))


if __name__ == "__main__":
    unittest.main()
