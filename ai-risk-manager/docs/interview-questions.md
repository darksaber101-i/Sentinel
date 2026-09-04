# Interview Preparation — Sentinel

---

## Project Pitches

**30-second pitch:**
"I built an end-to-end ML risk management platform that predicts whether an
e-commerce order will be returned before it's fulfilled. It uses a Gradient
Boosting model trained on 15,000 synthetic orders, served via FastAPI, with a
Next.js dashboard that shows explainable predictions, model performance metrics,
and a threshold simulator. The system demonstrates the full ML lifecycle from
data to deployment."

**2-minute pitch:**
"The problem is that e-commerce returns cost merchants 3–4× the original
shipping cost. I built a system that scores each incoming order with a risk
probability (0–100) so the ops team can prioritize manual reviews.

I generated a realistic synthetic dataset with 15,000 orders and trained three
classifiers — Logistic Regression, Random Forest, and XGBoost. The best model
is automatically selected by F1 score on a validation set; the test set was
never used during training.

The system exposes a FastAPI backend with a SQLite/PostgreSQL database that
stores orders and predictions. The Next.js frontend shows a risk dashboard,
an order table with filters, a model performance page with confusion matrices
and ROC curves, a threshold simulator showing the precision/recall trade-off,
and an AI assistant that answers questions about the data.

Every metric shown in the UI is computed from the real trained model — nothing is hardcoded."

---

## Machine Learning

**Q: Why did you choose binary classification?**
A: The target variable is binary — an order either returns (1) or doesn't (0).
Classification is the right tool. We also get a probability output which we
convert to a continuous risk score, which is more useful than a hard yes/no.

**Q: Why train multiple models?**
A: Different algorithms have different inductive biases. Logistic Regression
is a strong linear baseline. Random Forest handles non-linear interactions.
XGBoost is usually state-of-the-art on tabular data. By comparing all three,
we reduce the risk of a poor model choice and can discuss trade-offs.

**Q: What is precision vs recall?**
A: Precision = of all orders I flagged as risky, what fraction actually returned?
Recall = of all orders that actually returned, what fraction did I catch?
They trade off: a high-recall model flags more orders (catches more returns but
more false alarms). A high-precision model is more conservative.

**Q: Why is accuracy a misleading metric here?**
A: With 35% return rate, a model that predicts "no return" for every order
achieves 65% accuracy — but catches zero returns. Accuracy hides class imbalance.
F1 and PR-AUC capture the real performance.

**Q: What is ROC-AUC?**
A: The probability that a randomly chosen returned order gets a higher risk score
than a randomly chosen non-returned order. 1.0 = perfect, 0.5 = random.
It's threshold-independent, so useful for comparing models.

**Q: What is PR-AUC and why is it better than ROC-AUC for imbalanced data?**
A: PR-AUC is the area under the Precision-Recall curve. ROC-AUC can be
misleadingly high when the negative class is large (even a bad model gets
high TNR). PR-AUC focuses only on the positive class, making it more
informative when positives are rare.

**Q: What is a confusion matrix?**
A: A 2×2 table: True Negatives (correctly predicted as not-returned), False
Positives (wrongly flagged), False Negatives (missed returns), True Positives
(correctly caught returns). FN is the most costly error in a risk system.

**Q: Why did you use class_weight='balanced'?**
A: Our dataset is ~65% non-return, 35% return. Without balancing, the model
optimizes for the majority class (non-return) and underfits the minority.
`class_weight='balanced'` upweights minority class errors during training.

**Q: What is data leakage and how did you prevent it?**
A: Data leakage is when test-set information influences model training, giving
falsely optimistic metrics. Prevention: (1) split data before any fitting,
(2) fit scaler on train only and transform val/test, (3) never use test set
for model selection — only for final evaluation.

**Q: What is the train/validation/test split and why three sets?**
A: Train (70%) = the model learns parameters from this. Validation (15%) =
used for model selection and hyperparameter tuning. Test (15%) = held out
completely, evaluated only once at the end to report honest metrics.
Without a validation set, you'd tune on test data, which is leakage.

**Q: Why is the test set never touched during training?**
A: If you evaluate the test set multiple times and pick the model that
performs best on it, you're effectively fitting to the test set. The reported
test metrics would be optimistically biased. The test set is a simulation of
"unseen production data."

---

## Python / Pandas

**Q: What is pandas and why use it?**
A: pandas is a data manipulation library built on NumPy. It provides DataFrame
objects for tabular data, making it easy to load CSVs, filter rows, compute
group statistics, and prepare data for ML models.

**Q: How did you handle missing values?**
A: `days_since_last_order = -1` for new customers (no purchase history). We
replaced -1 with the median of valid values (days when previous orders existed).
This is median imputation — better than mean because it's robust to outliers.

**Q: What is one-hot encoding and why do it?**
A: ML models work with numbers, not strings. One-hot encoding converts a
categorical feature like `payment_method` into binary columns:
`payment_method_UPI`, `payment_method_COD`, etc. Each row has exactly one 1.
Tree models can handle it natively; LR needs it.

---

## Feature Engineering

**Q: How did you decide which features to engineer?**
A: Domain knowledge + correlation analysis. `value_per_item` captures per-unit
price rather than total — a single ₹10,000 item behaves differently from
10 × ₹1,000 items. `is_late_night_order` captures impulse purchasing behavior.
These are hypotheses that the model validates or discards through feature importance.

**Q: What is feature importance?**
A: The degree to which a feature contributes to model predictions. For tree
models (Random Forest, XGBoost), it's computed as average impurity decrease
across all splits using that feature. High importance = the model relies on
that feature a lot.

---

## Models

**Q: How does Logistic Regression work?**
A: It learns a linear combination of features → passes it through a sigmoid
function → outputs a probability. Despite the name, it's a classifier.
It works well when the decision boundary between classes is approximately linear.

**Q: How does Random Forest work?**
A: Builds many decision trees on random subsets of data and features (bagging),
then averages their predictions. The randomness reduces correlation between
trees and prevents overfitting.

**Q: How does XGBoost work?**
A: Gradient boosting builds trees sequentially, where each new tree corrects
the errors of the previous ones. XGBoost adds L1/L2 regularization and uses
second-order gradient information, making it faster and more accurate than
vanilla Gradient Boosting.

---

## Explainability

**Q: How do you explain individual predictions?**
A: Feature contribution = model_importance × |z_score|, where z_score measures
how far a feature value deviates from the training average. A feature that is
globally important AND extreme for this order gets a high contribution score.

**Q: What is SHAP?**
A: SHapley Additive exPlanations — a method from cooperative game theory.
It assigns each feature a contribution to the prediction such that contributions
sum to the difference between the prediction and the base rate. SHAP values
give the most theoretically rigorous local explanations.

---

## API / Backend

**Q: What is a REST API?**
A: An architectural style where endpoints correspond to resources (orders, metrics)
and HTTP verbs (GET, POST) correspond to actions (read, create). The backend
is stateless — all state is in the database.

**Q: Why FastAPI over Flask?**
A: FastAPI uses Python type hints and Pydantic for automatic request validation,
serialization, and interactive docs (/docs). It's also async-native and much
faster than Flask for I/O-bound workloads.

**Q: What is SQLAlchemy?**
A: An ORM (Object-Relational Mapper) that lets you define Python classes that
map to database tables. You write Python instead of SQL; SQLAlchemy generates
the SQL. It also handles database-agnostic code (SQLite dev / PostgreSQL prod).

---

## AI Assistant

**Q: Why not use the LLM as the prediction model?**
A: LLMs are not classifiers. They:
(1) Can't learn from tabular training data,
(2) Don't output calibrated probabilities,
(3) Can't be evaluated with precision/recall,
(4) Are 100-1000× slower and more expensive than a scikit-learn model,
(5) Would hallucinate predictions. The LLM is used only for natural language
explanation, grounded in real data — the ML model does the actual prediction.

**Q: How do you prevent the AI assistant from hallucinating?**
A: By injecting real application data into Claude's system prompt as structured
text. The model is instructed to answer ONLY using the provided data and
to explicitly say "data not available" if asked about something not in the context.

---

## Deployment

**Q: What is Docker?**
A: A containerization platform that packages an application with all its
dependencies into a portable image. "Works on my machine" becomes "works in any container."

**Q: How would you improve this with real data?**
A: (1) Use actual historical transaction data, (2) Add temporal features
(day of week, seasonality), (3) Retrain monthly with new data, (4) A/B test
threshold changes, (5) Track model drift with monitoring,
(6) Add customer segmentation, (7) Use LightGBM or CatBoost.

**Q: What is model drift?**
A: When real-world data distribution changes after deployment, causing model
performance to degrade. Monitoring: compare production prediction distribution
monthly against training distribution. Fix: retrain on recent data.
