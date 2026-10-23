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
| Making the model stable across months (drift) | Next |
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
3. **Get the data:** download the zip from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and unzip the 9 CSV files into `data/raw/`. With a Kaggle API token you can instead run `pip install -e ".[kaggle]"` and `python -m olist.download`.
4. **Build features and train:**

   ```bash
   python -m olist.features   # writes data/processed/orders_features.parquet
   python -m olist.train      # prints AUC for each model and writes reports/metrics_corrected_labels.json
   ```

5. **Run the tests:** `pytest`. They use a small fake dataset, so they work without the real data.

## How the data is turned into a model

**Label.** An order is late when it reaches the customer on a later day than promised. A naive label keeps only delivered orders, which drops late orders that never arrived and makes recent months look better than they were. The corrected label counts overdue, never-delivered orders as late, drops canceled orders, and keeps only orders whose promise date is before the end of the data, so every outcome is known. Compare both with `python -m olist.train --naive-labels`.

**Features (32), all known at the moment of purchase:**

- Order: number of items, sellers, total price, freight, freight-to-price ratio, weight, volume, category, payment type and instalments
- Promise: days promised, seller's ship-by deadline
- Geography: customer and seller state, coordinates, seller-to-customer distance, same-state flag
- Time: month, day of week, hour
- History: past late rate of the seller, the customer's state and the seller-to-customer route, and the seller's orders in the last 7 days. History only counts orders whose outcome was already known before this purchase, so nothing leaks from the future.

**Evaluation.** Each model is scored on a random split and on an out-of-time split (train on older months, test on the last 3). The out-of-time score is the honest one. Lateness in this data comes in waves, such as the early-2018 spike, so a model can look good on random orders and much worse on future months.

## Project layout

```
src/olist/
  config.py      paths and settings
  download.py    Kaggle download
  load.py        read raw CSVs with the right types
  labels.py      late label, including never-delivered orders
  features.py    one row per order, leak-free history features
  train.py       logistic regression vs LightGBM, random vs out-of-time
tests/           pytest suite on a small fake dataset
data/raw/        Kaggle CSVs (not committed)
```
