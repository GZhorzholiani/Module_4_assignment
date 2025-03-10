import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def model_xgboost(start_date, end_date, dataframe, dataframe_name, save_path=None, text_file_path=None):
    # Convert to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data up to the end date
    df_period = dataframe[dataframe["Date"] <= end_date].copy()

    # Features and target variable
    features = ['HospitalizedPatients', 'PatientsInIntensiveCare', 'TotalHospitalizedPatients',
                'HomeConfinement', 'CurrentPositiveCases', 'Healed', 'Dead']
    target = 'NewPositiveCases'

    # Create lag features (lags 1 to 7 days)
    for lag in range(1, 8):
        df_period[f'NewPositiveCases_lag_{lag}'] = df_period[target].shift(lag)

    # Drop missing values due to lagging
    df_period = df_period.dropna()

    # Define training data
    X = df_period[features + [f'NewPositiveCases_lag_{i}' for i in range(1, 8)]]
    y = df_period[target]

    # Train XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X, y)

    # Make predictions on training data
    y_pred = model.predict(X)

    # Evaluation
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Save the metrics and prediction results to a summary text file if provided
    if text_file_path:
        summary_text = f"Model Evaluation for {dataframe_name} ({start_date.date()} - {end_date.date()})\n"
        summary_text += f"XGBoost MSE: {mse}\n"
        summary_text += f"XGBoost RMSE: {rmse}\n"
        summary_text += f"XGBoost MAE: {mae}\n"
        summary_text += f"XGBoost R^2: {r2}\n"

        # Predict for the next 14 days
        future_dates = [end_date + pd.Timedelta(days=i) for i in range(1, 15)]
        future_preds = []

        last_known_values = df_period.iloc[-1][features].values.reshape(1, -1)
        last_lags = df_period.iloc[-1][[f'NewPositiveCases_lag_{i}' for i in range(1, 8)]].values.reshape(1, -1)

        for _ in range(14):
            future_input = np.hstack((last_known_values, last_lags))
            future_pred = model.predict(future_input.reshape(1, -1))[0]
            future_preds.append(future_pred)

            # Update lags for next prediction
            last_lags = np.roll(last_lags, shift=-1)
            last_lags[0, -1] = future_pred

        future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted New Positive Cases': np.round(future_preds, 0).astype(int)})
        summary_text += "\nPredictions for the next 14 days:\n"
        summary_text += future_predictions_df.to_string(index=False)

        # Write summary to file
        with open(text_file_path, "w") as f:
            f.write(summary_text)

    # Predict for the next 14 days and visualize
    future_preds = []
    future_dates = [end_date + pd.Timedelta(days=i) for i in range(1, 15)]

    last_known_values = df_period.iloc[-1][features].values.reshape(1, -1)
    last_lags = df_period.iloc[-1][[f'NewPositiveCases_lag_{i}' for i in range(1, 8)]].values.reshape(1, -1)

    for _ in range(14):
        future_input = np.hstack((last_known_values, last_lags))
        future_pred = model.predict(future_input.reshape(1, -1))[0]
        future_preds.append(future_pred)
        last_lags = np.roll(last_lags, shift=-1)
        last_lags[0, -1] = future_pred

    # Visualization
    plt.figure(figsize=(18, 9))
    plt.scatter(df_period["Date"], y, color='blue', label='Actual Values')
    plt.plot(df_period["Date"], y_pred, color='red', linestyle='dashed', label='Predicted Values')
    plt.plot(future_dates, future_preds, color='green', marker='o', linestyle='dashed', label='Future Predictions')
    plt.title(f"XGBoost {dataframe_name} Prediction for {start_date.date()} - {end_date.date()} and future 14-Day Forecast")
    plt.xlabel('Date')
    plt.ylabel('New Positive Cases')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    # Save the graph if a path is provided
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory
    else:
        plt.show()
