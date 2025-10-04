# NBA MVP Prediction Using Machine Learning

Predict the NBA's **Most Valuable Player (MVP)** from historical player statistics using multiple ML models with **explainability (SHAP)** and **temporal evaluation**.

> Notebook: `mvp_prediction.ipynb`

---

## Objectives
- Build a reproducible ML pipeline to predict MVP outcomes by season
- Compare models (Logistic Regression, Random Forest, XGBoost)
- Explain model decisions with **SHAP** and classic feature importance
- Validate **generalization over time** (train past → test recent seasons)

---

## Datasets Used
- `NBA_Dataset.csv` — season box score + advanced stats per player  
- `mvp_history.csv` — official MVP winners by season  
- `historical_RAPTOR_by_player.csv`, `modern_RAPTOR_by_player.csv` — FiveThirtyEight RAPTOR ratings

> Update file names/paths if yours differ.

---

## Tech Stack
- Python: pandas, numpy, scikit-learn, **xgboost**, **shap**
- Visualization: matplotlib, seaborn
- Notebook: Jupyter / VS Code

Install deps:
```bash
pip install -r requirements_mvp.txt
```

---

## Workflow
1) **Data Prep** — load, clean, join season stats with MVP labels; generate features (PER, WS, BPM, VORP, RAPTOR).  
2) **EDA** — target imbalance, feature correlations, trend plots.  
3) **Modeling** — baseline Logistic Regression; tree models (RF/XGBoost); cross‑validation & hyper‑parameters.  
4) **Evaluation** — Accuracy/ROC‑AUC, confusion matrix, season‑wise metrics.  
5) **Explainability** — SHAP summary & dependence plots; feature importance.  
6) **Temporal Validation** — train on 1980–YYYY, test on future seasons; rolling/hold‑out by season.  
7) **Conclusion** — which features drive MVP; when/why the model fails; next steps.

---

## Key Figures (export under `images/`)
- `images/roc_auc_comparison.png`
- `images/confusion_matrix_best_model.png`
- `images/feature_importance_rf.png`
- `images/feature_importance_xgb.png`
- `images/shap_summary_xgb.png`
- `images/shap_dependence_ws.png`
- `images/temporal_accuracy_by_season.png`

> Tip: save from the notebook with
```python
plt.savefig("images/<name>.png", bbox_inches="tight", dpi=150)
```

---

## Reproduce
```bash
# optional venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install
pip install -r requirements_mvp.txt

# run
jupyter lab
# open: mvp_prediction.ipynb
```

---

## Personal Contributions (example wording)
- Implemented **SHAP** for XGBoost & Random Forest; authored insights.
- Designed **temporal evaluation** across seasons.
- Wrote conclusions & documentation; added markdown to the notebook.
- Data prep & feature engineering for advanced metrics.

---

## Repository Structure
```
MVP-Prediction/
├── mvp_prediction.ipynb
├── requirements_mvp.txt
├── LICENSE
├── .gitignore
├── data/
│   ├── NBA_Dataset.csv
│   ├── mvp_history.csv
│   ├── historical_RAPTOR_by_player.csv
│   └── modern_RAPTOR_by_player.csv
└── images/
```
