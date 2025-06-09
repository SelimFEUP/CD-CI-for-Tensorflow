import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_utils import load_data, preprocess_data, create_sequences, split_data
import os

def test_model_evaluation():
    filepath = "./data/transformed_data.csv"
    data = load_data(filepath)
    #data.fillna(method='ffill', inplace=True)
    #data.interpolate(method='linear', inplace=True)
    data_scaled, _ = preprocess_data(data)

    input_steps, output_steps = 24, 24
    X, y = create_sequences(data_scaled, input_steps, output_steps)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Use a dummy model: just repeat the last input time step
    y_pred = X_test[:, -1:, :]  # shape (N, 1, features)
    y_pred = np.repeat(y_pred, output_steps, axis=1)  # shape (N, output_steps, features)

    # Reshape for MAE/RMSE
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_test.shape[-1])

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = mean_squared_error(y_test_flat, y_pred_flat, squared=False)

    print(f"\n🔥 MAE: {mae:.4f}")
    print(f"🔥 RMSE: {rmse:.4f}\n")

    # Save to models/metrics.txt
    os.makedirs("./models", exist_ok=True)
    with open("./models/metrics.txt", "w") as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    assert mae < 1.0
    assert rmse < 1.0

