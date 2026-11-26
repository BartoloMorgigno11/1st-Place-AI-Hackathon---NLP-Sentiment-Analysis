"""Public-data-free checks of the disclosed TF-IDF / logistic-regression pipeline."""
import importlib.util
import pathlib
import unittest

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_sentiment_model", ROOT / "scripts" / "train_sentiment_model.py"
)
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

    def test_model_learns_discriminative_features_without_contest_data(self):
        # This is a deliberately linearly separable algorithmic fixture,
        # not a claim of real-world generalisation or challenge accuracy.
        train = pd.Series([
            "excellent clean comfortable", "friendly clean lovely",
            "great helpful peaceful", "wonderful good pleasant",
            "awful dirty noisy", "terrible rude dirty",
            "bad broken unpleasant", "poor noisy disappointing",
        ])
        labels = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
        vectorizer, train_vec, _ = training.vectorize_text(train, train.iloc[:1], 100)
        model = training.train_model(train_vec, labels)
        predictions = model.predict(vectorizer.transform(pd.Series(["clean", "dirty"])))
        self.assertEqual(predictions.tolist(), [1, 0])

    def test_external_dataset_scoring_matches_direct_predictions(self):
        train = pd.Series([
            "excellent clean room", "friendly clean staff",
            "awful dirty room", "terrible dirty staff",
        ])
        labels = pd.Series([1, 1, 0, 0])
        vectorizer, train_vec, _ = training.vectorize_text(train, train.iloc[:1], 100)
        model = training.train_model(train_vec, labels)
        scored = pd.DataFrame({
            "Review": ["excellent clean", "awful dirty"],
            "Promotore": [1, 0],
        })
        original_preprocess = training.preprocess_dataframe
        try:
            # Offline test: isolate the evaluator contract without NLTK downloads.
            training.preprocess_dataframe = lambda frame: frame["Review"]
            evaluation = training.evaluate_dataset(model, vectorizer, scored)
        finally:
            training.preprocess_dataframe = original_preprocess
        from sklearn.metrics import accuracy_score
        expected = accuracy_score(
            scored["Promotore"], model.predict(vectorizer.transform(scored["Review"]))
        )
        self.assertEqual(evaluation, expected)

    def test_synthetic_fixture_has_balanced_unique_artificial_reviews(self):
        fixture_spec = importlib.util.spec_from_file_location(
            "synthetic_fixture", ROOT / "scripts" / "generate_synthetic_fixture.py"
        )
        fixture = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture)
        self.assertEqual(len(fixture.ROWS), 16)
        self.assertEqual(sum(label == 1 for _, label in fixture.ROWS), 8)
        self.assertEqual(sum(label == 0 for _, label in fixture.ROWS), 8)
        self.assertEqual(len({text for text, _ in fixture.ROWS}), len(fixture.ROWS))


if __name__ == "__main__":
    unittest.main()
