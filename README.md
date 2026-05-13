# Amazon-Recommendation-System
Data Science certification capstone: built and benchmarked 3 recommendation system approaches (rank-based, KNN collaborative filtering, SVD matrix factorization) on 7.8M+ Amazon ratings. Best model achieved RMSE 0.88 and F1@10 of 0.866 after GridSearchCV hyperparameter tuning.

A machine learning project that builds and compares three different recommendation system approaches on the **Amazon Electronics ratings dataset**. Built as a capstone project for my **Data Science certification**, the system predicts product ratings and generates personalized product recommendations for users using popularity-based, collaborative filtering, and matrix factorization techniques.

**Author:** Carolina Ochoa 

---

## Project Overview

In a world of ever-growing e-commerce catalogs, recommendation systems are essential for helping users discover relevant products and for helping businesses drive engagement. This project tackles that problem on a real-world dataset by:

1. Cleaning and preparing a very large, sparse interaction matrix.
2. Building a **baseline** version and an **optimized** (hyperparameter-tuned) version of three different recommendation algorithms.
3. Evaluating each model with appropriate ranking and rating-prediction metrics.
4. Comparing models to select the best performer.

**Final outcome:** The **Optimized SVD (Matrix Factorization)** model produced the lowest RMSE (0.8808) and the most balanced precision/recall, making it the best choice for this dataset.

---

## Dataset

- **Source:** Amazon product reviews — Electronics category.
- **Original size:** 7,824,482 ratings.
- **Columns:** `user_id`, `prod_id`, `rating`, `timestamp` (timestamp dropped).
- **Rating scale:** 1 to 5.
- **Note:** The dataset intentionally excludes product metadata and review text to prevent bias when building the model — recommendations are based purely on the rating signal.


---

## Tech Stack

**Language:** Python 3
**Environment:** Google Colab (chosen for compatibility with the `surprise` library)

**Core libraries:**
- `pandas` and `numpy` — data manipulation and numerical operations
- `matplotlib` and `seaborn` — exploratory data visualization
- `scikit-learn` — utility functions and metrics
- `scikit-surprise` — building, training, and evaluating recommendation models (`KNNBasic`, `SVD`, `Reader`, `Dataset`, `GridSearchCV`, `train_test_split`)
- `collections.defaultdict` — efficient grouping for precision/recall@k calculations

---

## Data Cleaning Process

The raw dataset was far too large and too sparse to be modeled directly — most users had rated only a handful of products, and many products had only a few ratings. To produce a meaningful and computationally feasible dataset, the following filtering steps were applied:

### Step 1 — Drop irrelevant columns
The `timestamp` column was dropped, as time is not used by the recommendation models in this project.

### Step 2 — Filter inactive users
Only users with **at least 50 ratings** were kept. The reasoning: collaborative filtering needs enough signal per user to find meaningful similarities; users with one or two ratings introduce noise without contributing useful patterns.

### Step 3 — Filter rarely-rated products
Only products with **at least 5 ratings** were kept. Products with very few ratings cannot be reliably recommended and replicate real-world e-commerce behavior — shoppers tend to trust products that have accumulated a baseline of reviews.

### Step 4 — Validate the cleaned dataset
- **Rows after cleanup:** 62,290 (down from ~7.8 million).
- **Unique users:** 1,540.
- **Unique products:** 5,689.
- **Missing values:** None.
- **Data types:** `user_id` (object), `prod_id` (object), `rating` (float).

This drastic reduction made the data dense enough for collaborative filtering while still preserving a realistic recommendation scenario.

---

## Exploratory Data Analysis (EDA)

Key findings from the EDA phase:

- **Rating distribution is heavily skewed toward 5 stars.** Roughly half of all ratings are 5, and about 27.8% are 4 — meaning ~78% of ratings are positive. Models must be evaluated carefully because predicting "high rating" is the naive baseline.
- **Mean rating:** 4.29 — confirming the positive skew.
- **Most active user** rated 295 products — still a tiny fraction of the 5,689 unique products available, meaning there is plenty of "unexplored" inventory for every user. This is exactly the gap a recommendation system is designed to fill.
- The user–item matrix is very sparse, which motivates the use of latent-factor methods (SVD) in addition to similarity-based methods.

---

## Models Implemented

For each collaborative filtering and matrix factorization model, I implemented **both a baseline and a hyperparameter-tuned version** to demonstrate the value of optimization.

### Model 1 — Rank-Based Recommendation System (Popularity Model)
A non-personalized baseline. Recommends the top-`n` products with the highest **average rating**, filtered by a minimum number of interactions (e.g. 50 or 100) so that obscure items with one perfect rating don't dominate the list. This model is the standard cold-start solution: it works for brand-new users with no history, but offers no personalization.

### Model 2 — Collaborative Filtering (KNN-based)

#### 2a. User–User Similarity
Finds users who are similar based on their rating patterns (using **cosine similarity**), then recommends products those similar users liked. Implemented with `KNNBasic` from the `surprise` library.

#### 2b. Item–Item Similarity
Finds products that are rated similarly across users, then recommends items similar to ones the target user already liked. Generally more stable than user–user filtering on sparse data because item profiles change more slowly than user profiles.

**Hyperparameter tuning** was performed with `GridSearchCV` over:
- `k` (max neighbors): 5, 10, 20, 30, 40, 50
- `min_k` (min neighbors): 1, 2, 3, 6, 9
- `sim_options.name`: `msd`, `cosine`, `pearson`, `pearson_baseline`
- `user_based`: True / False

### Model 3 — Model-Based Collaborative Filtering (Matrix Factorization with SVD)
The most powerful technique used in this project. **Singular Value Decomposition** learns latent features that represent users and items in a shared low-dimensional space. Unlike KNN, it doesn't depend on direct similarity computations and handles sparsity gracefully, which is why it ended up being the best performer.

**Hyperparameter tuning** for SVD with `GridSearchCV`:
- `n_epochs`: 10, 20, 30
- `lr_all` (learning rate): 0.001, 0.005, 0.01, 0.02
- `reg_all` (regularization): 0.05, 0.2, 0.4, 0.6

**Best parameters found:** `n_epochs=20`, `lr_all=0.01`, `reg_all=0.2`.

---

## Evaluation Metrics

Choosing the right metric matters as much as choosing the right model. This project uses a combination of rating-accuracy and ranking-quality metrics:

| Metric | What it measures | Why it matters |
|---|---|---|
| **RMSE** | Average prediction error vs. actual rating | Penalizes large mistakes; lower is better |
| **Precision@k** (k=10, threshold=3.5) | Of the top-k recommendations, how many are truly relevant | Avoids recommending products the user wouldn't enjoy |
| **Recall@k** | Of all relevant items, how many made it into the top-k | Avoids missing products the user would have liked |
| **F1@k** | Harmonic mean of Precision@k and Recall@k | Balances both concerns into a single score |

The relevance threshold was set at **3.5** — any rating at or above is considered a positive ("relevant") item.

---

## Results

| Model | RMSE | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|
| User–User KNN (baseline) | ~1.00 | 0.871 | 0.683 | 0.766 |
| User–User KNN (optimized) | Improved | Improved | Improved | Improved |
| Item–Item KNN (baseline) | 0.9950 | 0.838 | 0.845 | 0.841 |
| Item–Item KNN (optimized) | 0.9567 | 0.838 | 0.889 | 0.863 |
| SVD (baseline) | 0.8882 | 0.853 | 0.880 | 0.866 |
| **SVD (optimized) — Best** | **0.8808** | **0.854** | **0.878** | **0.866** |

**Winner:** Optimized SVD. It produced the lowest RMSE and the most balanced precision/recall combination, meaning its predictions are both accurate in magnitude and well-ranked.

---

## Sample Predictions

Two test cases were used throughout the project to compare models qualitatively:

- `user_id = "A3LDPF5FMB782Z"` who **had** interacted with `prod_id = "1400501466"` (true rating: 5).
- `user_id = "A34BZM6S9L7QI4"` who **had not** interacted with `prod_id = "1400501466"`.

The optimized SVD predicted **4.13** for the interacted user (close to the true 5) and **4.22** for the non-interacted user — both reasonable, neither overfit to the positive skew of the data.

A `get_recommendations()` helper function was also written to return the **top-N personalized product IDs** for any user using any trained algorithm.

---

## Project Structure

```
.
├── README.md                              <- You are here
├── RecommendationSystemsCarolina.ipynb    <- Main Jupyter notebook
├── RecommendationSystemsCarolina.html     <- Exported HTML view of the notebook
└── data/
    └── ratings_Electronics.csv            <- Raw dataset (download separately, see Dataset section)
```

---

## How to Run

1. **Clone the repository.**
```bash
   git clone https://github.com/<your-username>/amazon-recommendation-system.git
   cd amazon-recommendation-system
```

2. **Download the dataset.** Place `ratings_Electronics.csv` in the `data/` folder. The dataset is publicly available from the Amazon product data archive maintained by Julian McAuley (UCSD).

3. **Open the notebook.** This project was developed in **Google Colab** to avoid `surprise` installation issues that can occur on local Jupyter setups. Open `RecommendationSystemsCarolina.ipynb` in Colab and update the file path in the data-loading cell.

4. **Install dependencies** (if running locally):
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise
```

5. **Run all cells.** GridSearchCV steps can take several minutes — this is normal.

---

## Practices and Concepts Demonstrated

This project showcases the following data science practices, all of which are standard in production ML work:

- End-to-end ML pipeline: ingestion → cleaning → EDA → modeling → evaluation → comparison → conclusion.
- Defensive data cleaning with justification (every filter has a documented "why").
- Train/test splitting with a fixed `random_state` for reproducibility.
- Baseline-first methodology: every advanced model is compared to a simpler version.
- Hyperparameter tuning via `GridSearchCV` with cross-validation.
- Multi-metric evaluation (avoiding the trap of optimizing a single number).
- Sparse-matrix awareness: choosing algorithms that suit a sparse user–item matrix.
- Reproducibility: fixed random seeds, documented dependencies, exported HTML notebook.
- Honest result reporting — including cases where the optimized model only marginally improves on the baseline.

---

## Conclusion and Future Work

Both KNN-based and matrix factorization approaches produced usable recommendations, but **SVD with tuned hyperparameters was the strongest performer**. Because the rating distribution is skewed toward 5s, accurate top-N recommendations directly translate into more high-rating interactions — meaningful for any e-commerce business.

**Things I would improve next:**
- More aggressive hyperparameter tuning (wider grids, Bayesian optimization).
- Try additional `surprise` algorithms: `SVD++`, `NMF`, `BaselineOnly`, `CoClustering`.
- Hybrid recommender combining popularity for cold-start with SVD for warm users.
- Incorporate product metadata or review text (if business constraints allowed) for a content-based hybrid.
- Deploy the best model behind a small API (Flask / FastAPI) to demonstrate productionization.

---

## Reflections

This project was challenging but genuinely fun. Even though the dataset is sparse and only contains user IDs, product IDs, and ratings — no rich metadata — it was striking how much signal a recommendation system can extract from just those three columns. The initial dataset looked overwhelming at 7.8 million rows, but a well-reasoned cleanup made it manageable in minutes, which reinforced the value of disciplined preprocessing before reaching for fancy models.

---

## License

This project is released for educational and portfolio purposes.
