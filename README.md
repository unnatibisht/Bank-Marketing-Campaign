# 🏦 Bank Marketing Campaign: Predicting Term Deposit Subscriptions with Logistic Regression

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

An end-to-end classification project that predicts whether a bank customer will subscribe to a **term deposit** after a marketing phone call. It covers EDA, hypothesis-driven analysis, a leak-free preprocessing pipeline, model evaluation, and threshold tuning for an imbalanced target.

---

## 🎯 Project Objective

- Build a **Logistic Regression** model to predict whether a customer subscribes to a term deposit.
- Use customer profile, campaign details, previous contact history, and economic indicators as features.
- Help the bank **target the right customers** instead of calling everyone blindly.

**Target variable (`subscribed`, originally `y`):**

- `yes` → customer subscribed to a term deposit (encoded as `1`)
- `no` → customer did not subscribe (encoded as `0`)

> A term deposit is money placed with a bank for a fixed period (e.g. 6 months or 1 year) in exchange for interest.

---

## 📂 Dataset

- **File:** `bank-additional-full.csv` (semicolon-separated)
- **Source:** UCI Machine Learning Repository, Bank Marketing dataset (Moro et al., 2014)
- **Size:** 41,188 rows × 21 columns (41,176 rows after removing 12 duplicates)
- **Class balance:** ~88.7% `no` vs ~11.3% `yes` (highly imbalanced)
- **Missing values:** none as nulls, but several columns contain `'unknown'` categories

**Feature groups:**

| Group | Columns |
|---|---|
| Customer profile | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Current campaign | `contact`, `month`, `day_of_week`, `duration`, `campaign` |
| Previous campaign | `days_since_previous_contact` (`pdays`), `previous_contacts` (`previous`), `previous_campaign_outcome` (`poutcome`) |
| Economic context | `employment_variation_rate`, `consumer_price_index`, `consumer_confidence_index`, `euribor_3_month_rate`, `number_of_employees` |

Columns were renamed from the original dotted names (e.g. `emp.var.rate`) to readable snake_case names.

---

## 🧪 Hypotheses

The analysis was guided by hypotheses drawn from existing literature:

- **Age:** subscription rates differ between age groups.
- **Job & education:** subscription rates differ by occupation and education level.
- **Contact month:** some months produce more successful campaigns than others.
- **Number of contacts:** contact frequency is related to subscription success.
- **Previous outcome:** past campaign results are related to current subscription.
- **Economic conditions:** employment, prices, and interest rates influence subscription.

---

## 🔄 Workflow

- **1. Data loading and cleaning**
  - Loaded the CSV with `sep=";"`
  - Renamed columns for readability
  - Checked `info()`, descriptive statistics, nulls, and `'unknown'` counts
  - Removed 12 duplicate rows
- **2. Exploratory Data Analysis**
  - Numeric distributions (histograms) and categorical distributions (count plots)
  - Target distribution to confirm class imbalance
  - Subscription by age, job, contact type, contact month, education, and housing loan
  - Average call duration by job and subscription status
  - Euribor 3-month rate distribution
- **3. Train-test split**
  - 80/20 split, `stratify=y` to preserve class ratio, `random_state=42`
  - Train: 32,940 rows, Test: 8,236 rows
- **4. Preprocessing pipeline** (`ColumnTransformer`)
  - Numeric features → `StandardScaler`
  - Categorical features → `OneHotEncoder(handle_unknown="ignore")`
- **5. Modeling**
  - `Pipeline` combining the preprocessor and `LogisticRegression(max_iter=1000, random_state=42)`
  - Preprocessing is fit on training data only, which avoids data leakage from the test set
- **6. Evaluation**
  - Accuracy, precision, recall, F1, confusion matrix, ROC curve, and AUC
- **7. Threshold tuning**
  - Selected the optimal threshold using **Youden's J statistic** (TPR − FPR) from the ROC curve

---

## 📊 Results

### Default threshold (0.50)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No (0) | 0.93 | 0.97 | 0.95 |
| Yes (1) | 0.65 | 0.42 | 0.51 |

- **Accuracy:** 90.95%
- High accuracy here is misleading: with ~89% of customers in the `no` class, the model misses most actual subscribers (recall of 0.42 for `yes`).

### Optimized threshold (0.099, via Youden's J)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No (0) | 0.99 | 0.84 | 0.91 |
| Yes (1) | 0.43 | 0.92 | 0.58 |

- **Accuracy:** 85%
- **Recall for `yes` rose from 0.42 to 0.92**, and F1 for `yes` improved from 0.51 to 0.58.
- Trade-off: precision for `yes` fell from 0.65 to 0.43, meaning more calls go to customers who won't subscribe.

---

## 💡 Key Takeaways

- Lowering the threshold to **0.099** lets the model catch **92% of customers who actually subscribe**, at an overall accuracy of 85%.
- This lets the bank prioritise likely subscribers rather than contacting everyone.
- Call duration and previous campaign history stood out as strong signals of who would subscribe.
- On imbalanced data, **accuracy alone is a poor metric**; recall, precision, F1, and ROC-AUC give a more honest picture.

---

## 🛠 Tech Stack

- **Language:** Python
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistics:** SciPy
- **Machine Learning:** scikit-learn (`Pipeline`, `ColumnTransformer`, `LogisticRegression`)

---

## 📚 References

- Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems.
- Nechita et al. (2024), cited in the notebook for job, education, and previous-outcome effects.
- Dataset: [UCI Machine Learning Repository, Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

---

## 👩‍💻 Author

**Unnati Bisht**

- 🌐 Portfolio: [unnati-website.netlify.app](https://unnati-website.netlify.app)
- 💼 LinkedIn: [linkedin.com/in/unnati-bisht-039356315](https://linkedin.com/in/unnati-bisht-039356315)
- 🐙 GitHub: [github.com/unnatibisht](https://github.com/unnatibisht)
- 📧 Email: ubishtt@gmail.com
