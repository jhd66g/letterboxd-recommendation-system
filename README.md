# Letterboxd Movie Recommender

A hybrid collaborative‑filtering and content‑based recommendation system for movies, leveraging Letterboxd review data and TMDb metadata.

---

## Overview

This repository provides:

1. **Data pipeline** to collect and preprocess review and metadata.
2. **Recommendation engine** that trains a LightFM hybrid CF–CB model.
3. **API** to serve top‑N movie recommendations for any user.
4. **Test and demo scripts** to validate functionality.

---

## Data Considerations

- **Users & Reviews**

  - Target: 200–1,000 high‑quality users, each with ≥1,000 reviews.
  - Estimates:
    - 200 users → \~400,000 reviews → \~15,000 unique films
    - 500 users → \~1,000,000 reviews → \~30,000 unique films

- **Feature Engineering**

  - **Bucket & one‑hot** for year, runtime, budget, revenue, popularity tiers.
  - **Multi‑hot** for genres, top directors, top cast, top languages.
  - **TF–IDF** for keywords (and overview, if enabled), with optional dimensionality reduction.

---

## Data Pipeline

Scripts in `pipeline/` automate data collection and feature engineering:

1. `get_users.py`

   - Fetches top‑N popular Letterboxd usernames.

2. `scraper.py`

   - Scrapes each user’s reviews (title, year, rating).
   - Only captures reviews with explicit ratings.

3. `unique_movie_extractor.py`

   - Filters to unique movie title/year pairs reviewed ≥5 times.

4. `tmdb_fetcher.py`

   - Resolves each movie to TMDb ID and fetches metadata: release date, runtime, directors, cast, genres, original language, overview, keywords, streaming services, poster path, budget, revenue, popularity.

5. `data_cleaner.py`

    - Adds TMDB IDs to review data from metadata.
    - Fills in missing years in review data.

6. `feature_preparation.py`

   - Executes feature engineering and writes `data/final_features*.csv`:
     - `tmdb_id`
     - Year → bucket by decade (one-hot; `Pre-1950`, `1950s`, …, `2020s`)
     - Runtime → quartile bins e.g. `(0,87]`, `(87,98]`, `(98,111]`, `(111,∞)`, one-hot
     - Directors → multi-hot top 200 directors; rest mapped to `Other`
     - Cast → multi-hot top 500 cast members; rest mapped to `Other`
     - Genres → multi-hot for each of 19 genres
     - Keywords → TF–IDF transform (full set)
     - Budget → log1p transform; bucket; one-hot
     - Revenue → log1p transform; bucket; one-hot
     - Original language → one-hot for top 10 languages; rest → `Other`
     - Popularity → bucket (`low <1`, `medium 1–5`, `high >5`); one-hot

6. `run_data_pipeline.sh`. ` `

   - Executes full pipeline: `get_users.py` → `scraper.py` → `unique_movies_extractor.py` → `tmdb_fetcher.py` → `data_cleaner.py` → `feature_preparation.py`.
   - Archives previous runs; outputs timestamped `final_reviews*.csv`, `final_metadata*.csv`,  and `final_features*.csv` in `data/`.

---
## Model Design

A hybrid LightFM model learns one embedding per user and per item, where each item embedding is the sum of:

- **ID embedding (CF)**  
  – Learned from the user×item interaction matrix of explicit ratings.  
  – **Boost** this signal by scaling the “item_id” feature column by `CF_WEIGHT` (e.g. 2.0) or by duplicating that column.

- **Feature embeddings (CB)**  
  – Learned from metadata features: year-bins, runtime-bins, genres, directors, cast, keywords, budget, revenue, language, popularity.  
  – **Attenuate** these by scaling all metadata columns by `CB_WEIGHT` (e.g. 0.5) so they don’t overwhelm the CF component.

**Balancing CF vs. CB**  
In `feature_preparation.py`, after building the sparse feature matrix:
```python
features[:, item_id_index] *= CF_WEIGHT
features[:, metadata_indices] *= CB_WEIGHT
```

**Ablation Testing**

Support three modes to measure CF vs. CB contribution:
1. **CF-only**: zero out all metadata columns.
2. **CB-only**: zero out the “item_id” column.
3. **Hybrid**: include both, with scaling applied.

Compare Precision@K, Recall@K, NDCG across modes to verify the CF signal is retained.

**Hyperparameters & Training**

- Loss: `warp` (or `bpr`)
- Components: 40–100 latent factors
- Regularization: L2 (`item_alpha`, `user_alpha`)
- Train/test split: 80/20 (time‑aware if possible)
- Metrics: Precision\@K, Recall\@K, NDCG, AUC

**Output**

- Saved model file: `models/lightfm_{timestamp}.npz`

---

## Recommendation Engine

**File**: `recommendation_engine.py`

A production-grade pipeline that assembles data, trains the hybrid model, and serves recommendations in clear, repeatable steps:

1. **Load preprocessed data**  
   - Read `final_reviews*.csv` into a user×item interaction DataFrame.  
   - Read `final_features*.csv` into a movie×feature DataFrame.  
   - Load sparse matrices for interactions and item features from `.npz` files.

2. **Scale and adjust features**  
   - Multiply the “item_id” column by `CF_WEIGHT` to boost collaborative signal.  
   - Multiply all metadata columns by `CB_WEIGHT` to attenuate content features.  
   - (Optional) Duplicate the “item_id” column if further boosting is needed.

3. **Select ablation mode**  
   - **CF-only**: zero out all metadata columns, retaining only the ID feature.  
   - **CB-only**: zero out the “item_id” column, training on metadata alone.  
   - **Hybrid**: use the full scaled matrix.

4. **Initialize the LightFM model**  
   - Choose loss (`warp` or `bpr`), latent dimensionality (40–100), and regularization (`user_alpha`, `item_alpha`).  
   - Set a fixed random seed for reproducibility.

5. **Train or load existing weights**  
   - If a saved model exists for the selected mode, load it.  
   - Otherwise, fit the model on the interaction matrix with the prepared item features, then save the resulting weights under `models/lightfm_{mode}_{timestamp}.npz`.

6. **Evaluate performance**  
   - On a held-out test split, compute ranking metrics (Precision@K, Recall@K, NDCG) for the chosen mode.  
   - Log or compare these metrics across CF-only, CB-only, and hybrid runs to confirm the CF component is sufficiently emphasized.

7. **Generate top-N recommendations**  
   - For a given user ID, predict scores against all items using the trained model and item features.  
   - Sort by descending score and select the top N item indices.  
   - Map those indices back to movie metadata (title, year, genres, etc.) and include the predicted score.

8. **Expose interface**  
   - Wrap the above steps in functions or a CLI flag (`--mode`) so that the engine can be invoked for training, evaluation, or recommendation generation in a repeatable, automated fashion.

**Core Modes**
- `--mode {hybrid, cf_only, cb_only}` *(default: `hybrid`)*  
  Selects how item features are constructed:
  - **hybrid**: combines scaled CB features with a boosted CF identity feature.
  - **cf_only**: uses only the CF identity feature (ablation).
  - **cb_only**: uses only metadata features (ablation).

**Actions**
- `--train`  
  Train a model for the selected `--mode`. Saves to `models/lightfm_{mode}_{timestamp}.npz`.
- `--evaluate`  
  Evaluate the current model for the selected `--mode` and print metrics (Precision@K, Recall@K, NDCG) as JSON.
- `--recommend USERNAME`  
  Generate recommendations for the given Letterboxd username.

> You can combine actions, e.g. `--train --evaluate` trains then evaluates.

**I/O Paths**
- `--reviews_file PATH`  
  Path to `final_reviews_*.csv`. If omitted, the engine auto-detects the most recent matching file.
- `--features_file PATH`  
  Path to `final_features_*.csv`. If omitted, the engine auto-detects the most recent matching file.
- `--model_file PATH`  
  Load an existing saved model. If provided, this overrides training unless you explicitly pass `--train`.

**Training Params**
- `--epochs INT` *(default: `10`)*  
  Number of training epochs for `--train`.

**Recommendation Params**
- `--top_n INT` *(default: `25`)*  
  Number of items to return for `--recommend`.

**Example: train a hybrid model for 30 epochs**
```bash
python recommendation_engine.py --mode hybrid --train --epochs 30
```
---

## API

**File**: `api.py`

A FastAPI service that wraps the recommendation engine. It auto-loads the latest `final_reviews_*`, `final_metadata_*`, and `final_features_*` files, loads the latest **hybrid** model if available (otherwise trains one), and **automatically scrapes and adds new users** on first request (quick retrain with a small number of epochs).


- Takes a `--user` (Letterboxd username) and checks if the user is already in the reviews database.
- If the user **exists** in the database: reuse data; if `--epochs` is not provided, reuse the most recent model (no training).
- If the user is **not** in the database: run a modified data pipeline to scrape reviews, map TMDb IDs and years using the metadata CSV, drop reviews not found in metadata, append valid reviews to `final_reviews*.csv`, then (if new data was added) train and recommend.
- For each recommended TMDb ID, return an object with the display metadata.

**CLI**
```bash
python api.py --mode MODE --epochs EPOCHS --user USERNAME --top_n N 
```

**Run the server**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Endpoints**
- `GET ` — health check

- `GET /user/{username}/status` — returns whether `username` exists in the dataset

- `GET /recommendations/{username}?top_n=25` — returns top-N recommendations
  - If `username` is **new**: scrapes reviews, matches to TMDb via the latest metadata CSV, appends to `final_reviews_{timestamp}`.csv, reinitializes + quick-retrain the hybrid model to include the new user, then returns recommendations.
  - If `username` exists: uses the currently loaded model and returns recommendations.

- `GET /recommendations/{username}?top_n=25&epochs=20` - returns topN recommendations with t epochs for training.

**Steps**
1. Parse `--mode`, `--epochs`, `--user`, `--top_n`.

2. Check if `USERNAME` is present in `final_reviews*.csv`.

3. If not present:
    - Verify the Letterboxd account exists.
      - Return error message if not found. 
    - Scrape reviews for `USERNAME`.
    - Keep only reviews whose films are present in the metadata CSV.
    - Append new reviews to `final_reviews*.csv`.
    - Run:
    ```bash
    python recommendation_engine.py --mode MODE --train --epochs EPOCHS --recommend USERNAME --top_n N
    ```

4. If already present and `--epochs` not provided:
    - Use the most recent saved model (no `--train`).
    - Generate recommendations with the latest model.

5. Write results to `data/username_recommendations_top{N}_{timestamp}.json`.

**Output**
- JSON ordered by predicted score, containing one object per recommended movie with fields.
- Example: 
```json
    [
      {
        "rank": 1,
        "tmdb_id": 12345,
        "title": "Example Movie",
        "year": 2020,
        "score": 0.87,
        "runtime": 98,
        "directors": ["Director A"],
        "cast": ["Actor X", "Actor Y"],
        "genres": ["Drama", "Thriller"],
        "language": "en",
        "overview": "Movie summary...",
        "streaming_services": ["Netflix", "Hulu"],
        "budget": 20000000,
        "revenue": 150000000
      }
    ]
```
---

## Test

**File**: `test.py`

Batch-runs the API for several usernames and reports results; also runs an evaluation pass.

**Usernames**
- schaffrillas
- davidehrlich
- kurstboy
- jaaackd
- aidandking08

**CLI**
```bash
python test.py --mode MODE
```
- default `MODE` = `hybrid`.

**Steps**

1. For each username above, call:
  ```bash
  python api.py --mode MODE --epochs 10 --user USERNAME --top_n 5
  ```

2. Display, for each username:
    - Whether the user is in the dataset or newly added.
    - Top 5 movies (rank, title, year, predicted score).
    - If the movie has already been reviewed by the user.

3. Run evaluation:
  ```bash
  echo "Evaluating for 30 epochs …"
  python recommendation_engine.py --mode MODE --train --epochs 30 --evaluate
  ```

4. Display aggregate evaluation metrics.

**Output**
- Hides all output from `api.py` and `recommendation_engine.py`.
- Prints:
  - Username
  - In-dataset statuses
  - Top-5 list
  - Final evaluation metrics

---

## Demo

**File**: `demo.py`

Runs the API for a **single** user and prints a human-readable recommendation report.

**CLI**
```bash
python demo.py --mode MODE --epochs EPOCHS --user USERNAME --top_n N
```
- defaults: `MODE` = `hybrid`, `EPOCHS` = `30`, `TOP_N` = `25`.
- `USERNAME` is required.

**Behavior**
- Internally calls:
  ```bash
  python api.py --mode MODE --epochs EPOCHS --user USERNAME --top_n N
  ```
- Hides all output from `api.py` and `recommendation_engine.py`. 
- Prints: 
  - Username
    - Top N movies with:
    - Rank, title, year, predicted score
    - Genres, director(s), cast, original language
    - Budget, revenue, runtime
    - Overview
    - Streaming services
---

## Project Structure

```
.
├── data/
│   ├── final_reviews*.csv
│   └── final_metadata*.csv
│   └── final_features*.csv
├── pipeline/
│   ├── get_users.py
│   ├── scraper.py
│   ├── unique_movies_extractor.py
│   ├── tmdb_fetcher.py
│   ├── data_cleaner.py
│   ├── feature_preparation.py
│   └── run_data_pipeline.sh
├── recommendation_engine.py
├── api.py
├── test.py
├── demo.py
├── models/
│   └── lightfm_<timestamp>.npz
└── README.md
```

