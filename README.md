# Randstad & MKERS Sentiment Analysis Challenge

## Overview
This repository contains the winning solution we built during the Randstad & MKERS AI Hackathon at Politecnico di Bari. The task was to deliver the most accurate sentiment analysis model in the shortest possible time using real hotel reviews. By focusing on a lean preprocessing pipeline and a well-regularised logistic regression classifier, we achieved a 96% accuracy score that secured first place.

Alongside the model, we provide a geomap visualisation that showcases the average reviewer score by country across Europe.

## Highlights
- Hackathon winner: Delivered the highest accuracy (96 percent) in record time.
- Simple and effective: TF-IDF features combined with a class-balanced logistic regression.
- Insights at a glance: European geomap of aggregated reviewer sentiment.
- Reproducible scripts: Run the full pipeline without opening the notebooks.

## Repository Structure
```
Geomap/
  geomap.ipynb
Reviews/
  logistic_regression.ipynb
map/
  ne_110m_admin_0_countries.*  (Natural Earth shapefile bundle)
scripts/
  generate_geomap.py
  train_sentiment_model.py
requirements.txt
README.md
README_ITA.md
```

## Getting Started
1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Supply data that you are authorised to use. The original hackathon CSVs are deliberately not distributed in this public portfolio edition. See [Data setup](#data-setup).
4. Train and evaluate the sentiment model:
   ```bash
   python scripts/train_sentiment_model.py --train-path path/to/training.csv
   ```
   The script downloads the required NLTK stopwords on first run. Results are printed to the console and the ROC curve plus confusion matrix are saved in the `outputs/` directory.
5. Generate the European geomap:
   ```bash
   python scripts/generate_geomap.py --data-path path/to/city_scores.csv
   ```
   The resulting figure is saved in `outputs/european_sentiment_map.png`. Add `--show` to display the map interactively.

## Customising the Pipelines
- Provide a different dataset using `--train-path path/to/data.csv`.
- Evaluate on another labelled dataset with `--extra-path path/to/extra.csv`.
- Change TF-IDF dimensionality via `--max-features`.
- Adjust the geomap colour range using `--vmin` and `--vmax`.

Run `python scripts/train_sentiment_model.py --help` or `python scripts/generate_geomap.py --help` to see all available options.

## Data setup

The contest datasets are not included. Do not add them to a fork or public copy unless you have explicit redistribution permission from their owner.

- The sentiment training CSV must include a free-form `Review` column and a binary `Promotore` label column. Pass it with `--train-path`; optionally pass a separate labelled set with `--extra-path`.
- The geomap CSV must include `City_Nation` (for example, `Bari, Italy`) and numeric `Reviewer_Score` columns. Pass it with `--data-path`.
- The shapefile under `map/` is sourced from Natural Earth (public domain).

## Outputs
- `outputs/roc_curve.png` - ROC curve on the validation split.
- `outputs/confusion_matrix.png` - 2x2 confusion matrix with class labels.
- `outputs/european_sentiment_map.png` - Heat map of average reviewer score per European country.

## Version Control Notes
- `.gitignore` keeps build artefacts, virtual environments, generated outputs and contest CSVs out of version control.

## License
The review datasets and derived artefacts belong to the hackathon organisers and participating team. The Natural Earth shapefiles are in the public domain. Unless stated otherwise, the code in this repository is released under the MIT License.
