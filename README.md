# Olist Delivery Predictor

Predicts late deliveries at checkout on 99k real Olist e-commerce orders, with LightGBM, SHAP explanations, a delivery-promise simulator and a Streamlit dashboard.

## The problem

[Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) is a Brazilian online marketplace. When a customer buys something, the site promises a delivery date. About 7% of orders arrive after that date, and those customers leave much worse reviews.

The goal is to predict, at the moment someone clicks "buy", whether their order will be late, so the business can act early: warn the customer, pick a faster carrier, or give a longer promise date.

## What's in the project

| Part | Status |
| --- | --- |
| Data download and order-level feature table | Done |
| Corrected late label (counts never-delivered orders) | Done |
| Baseline models: logistic regression vs LightGBM, random vs out-of-time test | Done |
| Drift report, recent-conditions features, month-by-month backtest | Done |
| Streamlit dashboard | Planned |
| Delivery-promise simulator | Planned |
| Chat assistant over the data | Later |

## Run it in VS Code

1. **Clone:** open the Command Palette (Ctrl/Cmd+Shift+P), choose **Git: Clone**, paste `https://github.com/Adi3514/olist-delivery-predictor`.
2. **Create a Python environment** in the VS Code terminal:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

   When VS Code asks, pick `.venv` as the interpreter.

   **Mac only:** LightGBM needs the OpenMP library. If `import lightgbm` fails with `Library not loaded: libomp.dylib`, run `brew install libomp`, or with conda, `conda install -c conda-forge llvm-openmp`.
3. **Get the data:** download the zip from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and unzip the 9 CSV files into `data/raw/`. With a Kaggle API token you can instead run `pip install -e ".[kaggle]"` and `python -m olist.download`.
4. **Build features and train:**

   ```bash
   python -m olist.features   # writes data/processed/orders_features.parquet and prints late orders by source
   python -m olist.drift      # late rate by month and the features that shift over time
   python -m olist.train      # compares 3 models on random, last-3-months and month-by-month tests
   ```

5. **Run the tests:** `pytest`. They use a small fake dataset, so they work without the real data.

## How the data is turned into a model

**Label.** An order is late when it reaches the customer on a later day than promised. A naive label keeps only delivered orders, which drops late orders that never arrived and makes recent months look better than they were. The corrected label (the default):

- counts orders that were handed to the carrier but never delivered, once their promise date has passed, as late (`lost_in_transit`);
- drops orders that never reached the carrier (stuck in "processing" or "invoiced"), because those are seller fulfilment failures, closer to cancellations than to slow deliveries;
- drops canceled and unavailable orders;
- keeps only orders whose promise date is before the end of the data, so every outcome is known.

Compare the three versions with `--label-mode`:

| Mode | Never-delivered, shipped | Never shipped |
| --- | --- | --- |
| `corrected` (default) | late | dropped |
| `corrected_all` | late | late |
| `naive` | dropped | dropped |

```bash
python -m olist.train --label-mode corrected_all
python -m olist.train --label-mode naive
```

**Features (37), all known at the moment of purchase:**

- Order: number of items, sellers, total price, freight, freight-to-price ratio, weight, volume, category, payment type and instalments
- Promise: days promised, seller's ship-by deadline
- Geography: customer and seller state, coordinates, seller-to-customer distance, same-state flag
- Time: month, day of week, hour
- History: past late rate of the seller, the customer's state and the seller-to-customer route, and the seller's orders in the last 7 days. History only counts orders whose outcome was already known before this purchase, so nothing leaks from the future.
- Recent conditions: late rate and average delivery time over the last 30 days, overall and for the customer's state, plus the promise slack (promised days minus how long deliveries to that state are taking right now). These let the model notice a wave as it starts.

**Evaluation.** Each model is scored three ways:

- **Random split:** test orders mixed in from all months. Optimistic.
- **Last 3 months:** train on older months, test on the newest three.
- **Backtest:** one test per month over the last 6 months, each time training on everything known before that month. The mean AUC is the headline number, and the worst month shows how badly a model can slip during a wave.

Training sets only contain orders whose outcome was known before the test period starts, as it would be in production. Lateness in this data comes in waves, such as the early-2018 spike, so a model can look good on random orders and much worse on future months.

**Models compared:**

| Model | What it is |
| --- | --- |
| `logistic_regression` | Simple baseline |
| `lightgbm` | Gradient-boosted trees on all features |
| `lightgbm_stable` | Smaller, more regularized trees; leaves out calendar-like features (month, raw coordinates, running order counts); weights recent orders more (half-life 120 days) |

**Drift report.** `python -m olist.drift` trains a model to tell old orders from recent ones. If it can, the features it leans on are the ones that drift.

## Project layout

```
src/olist/
  config.py      paths and settings
  download.py    Kaggle download
  load.py        read raw CSVs with the right types
  labels.py      late label, including never-delivered orders
  features.py    one row per order, leak-free history and recent-conditions features
  drift.py       late rate by month and adversarial validation
  train.py       3 models on random, last-3-months and month-by-month tests
tests/           pytest suite on a small fake dataset
data/raw/        Kaggle CSVs (not committed)
```
