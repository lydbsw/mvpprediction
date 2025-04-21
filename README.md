# NBA MVP Prediction

A machine learning project to predict NBA Most Valuable Player (MVP) award winners using historical player statistics and advanced metrics.

## Project Overview
This project demonstrates how to build and evaluate machine learning models that can predict NBA MVP winners based on player performance metrics, team success factors, and advanced analytics. By analyzing historical data patterns, we can identify the key factors that influence MVP voting and create predictive models to forecast future winners.

## Data Sources
- `mvp_history.csv`: Historical records of NBA MVP winners from 1956 to present  
- `NBA_Dataset.csv`: Comprehensive player statistics including traditional box‑score stats and advanced metrics  
- `historical_RAPTOR_by_player.csv`: FiveThirtyEight’s RAPTOR ratings for NBA players from earlier eras  
- `modern_RAPTOR_by_player.csv`: FiveThirtyEight’s RAPTOR ratings for current NBA players  

## Methodology
1. **Data Preparation**  
   - Merging multiple datasets  
   - Creating target variables  
   - Handling missing values  
2. **Feature Engineering**  
   - Selecting the most relevant statistical features for MVP prediction  
3. **Model Training**  
   - Building multiple classification models:  
     - Logistic Regression  
     - Random Forest Classifier  
     - XGBoost Classifier  
4. **Model Evaluation**  
   - Comparing models using classification metrics, ROC curves, and cross‑validation  
5. **Interpretability Analysis**  
   - Visualizing feature importance  
   - Using SHAP values to understand model decisions  
6. **Temporal Validation**  
   - Testing how well models trained on historical data predict recent MVPs  

## Key Features
- **Feature Selection**: Identifies the most important statistical metrics for MVP prediction  
- **Model Comparison**: Evaluates different algorithms to find the best approach  
- **Visualization**: Includes ROC curves, feature importance plots, and SHAP value analysis  
- **Time‑Based Analysis**: Examines how prediction accuracy changes across different NBA eras  
