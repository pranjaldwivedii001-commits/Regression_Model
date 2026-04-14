# Regression_Model
# Wind Turbine Power Output Forecasting

## Project Overview

This project aims to develop a robust machine learning model for accurately forecasting power output from individual wind turbines. The unpredictability of power output can lead to inefficiencies in grid planning, reduced profitability, and increased maintenance costs. An accurate prediction model is crucial for optimizing grid integration, improving operational efficiency, and reducing downtime.

## Objective

The primary objective is to develop a machine learning model that utilizes historical turbine and environmental data to predict the `Target` variable (expected power output) from wind turbines.

## Data Overview

The dataset (`train.csv`) contains historical operational data from various wind turbines, including timestamp, active power, ambient temperature, generator speed, wind speed, and other relevant environmental and mechanical parameters.

- **Shape:** (7575 rows, 16 columns)
- **Key Features:**
    - `timestamp`: Date and time of the reading.
    - `active_power_calculated_by_converter`: Active power output.
    - `ambient_temperature`: Surrounding air temperature.
    - `generator_speed`: Speed of the generator.
    - `wind_speed_raw`: Raw wind speed measurement.
    - `turbine_id`: Identifier for each unique turbine.
    - `Target`: The target variable for prediction (e.g., expected power output).

## Methodology

### 1. Data Cleaning and Preprocessing

- **Timestamp Conversion:** The `timestamp` column was converted to datetime objects.
- **Feature Engineering:** New time-based features (`hour`, `day_of_week`, `month`, `year`) and `wind_power_density` were extracted from the `timestamp` and `wind_speed_raw` features.
- **Missing Value Handling:** `SimpleImputer` was used for both numerical (mean strategy) and categorical (most frequent strategy) features.
- **Feature Scaling:** Numerical features were scaled using `StandardScaler`.
- **Categorical Encoding:** `turbine_id` was one-hot encoded using `OneHotEncoder`.

### 2. Feature Selection

Irrelevant columns such as `timestamp`, `active_power_raw`, and `reactive_power` were dropped from the feature set `X`.

### 3. Data Splitting

The dataset was split into training and testing sets with an 80/20 ratio (`random_state=42`).

### 4. Model Training and Evaluation

Several regression models were trained and evaluated based on Mean Squared Error (MSE) and R-squared (R2):

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Support Vector Regressor (SVR)
- Random Forest Regressor
- XGBoost Regressor

### 5. Hyperparameter Tuning (Random Forest)

RandomizedSearchCV was employed to fine-tune the hyperparameters of the Random Forest Regressor, searching across a wide distribution of parameters for `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf`, and `bootstrap`.

## Results and Model Comparison

| Model                               | MSE    | R-squared (R2) |
| :---------------------------------- | :----- | :------------- |
| Linear Regression                   | 5.07   | 0.35           |
| Ridge Regressor                     | 5.07   | 0.35           |
| Lasso Regressor                     | 7.51   | 0.04           |
| ElasticNet Regressor                | 7.01   | 0.11           |
| Support Vector Regressor (SVR)      | 4.76   | 0.39           |
| XGBoost Regressor                   | 4.51   | 0.42           |
| Original Random Forest Regressor    | 4.48   | 0.43           |
| **Tuned Random Forest Regressor**   | **3.91** | **0.48**         |

The **Tuned Random Forest Regressor** demonstrated the best performance, achieving the lowest Mean Squared Error (3.91) and the highest R-squared value (0.48), indicating it explains the most variance in the target variable.


<img width="442" height="267" alt="Screenshot (100)" src="https://github.com/user-attachments/assets/f4dff97c-e605-4a7f-831b-0d350f618930" />


## Visualizations

Key visualizations include:
- Box plot of Target vs. Turbine ID.
- Scatter plots of Target vs. Wind Speed, Ambient Temperature, and Generator Speed.
- Heatmap of the correlation matrix for numerical features.
- Actual vs. Predicted Target Values for the best-performing model.

## Conclusion

The project successfully developed and optimized a machine learning model for wind turbine power output forecasting. The Tuned Random Forest Regressor significantly outperformed other models, providing a more accurate prediction capability, which can aid in better grid planning and operational efficiency.
