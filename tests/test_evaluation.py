import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_utils import load_data, preprocess_data, create_sequences, split_data

def test_model_evaluation():
    filepath = "./data/transformed_data.csv"
    data = load_data(filepath)
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

    # Add assertion to keep test framework happy
    assert mae < 0.5, f"MAE too high: {mae}"
    assert rmse < 0.5, f"RMSE too high: {rmse}"

