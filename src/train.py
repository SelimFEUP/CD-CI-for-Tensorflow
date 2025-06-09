import numpy as np
from data_utils import load_data, preprocess_data, create_sequences, split_data
from model import build_model
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def main():
    filepath = 'data/transformed_data.csv'
    raw_data = load_data(filepath)
    scaled_data, scaler = preprocess_data(raw_data)

    input_steps = 24
    output_steps = 24

    X, y = create_sequences(scaled_data, input_steps, output_steps)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    model = build_model(input_shape=X_train.shape[1:], output_shape=y_train.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64)
    
    # Predict and reshape
    y_pred = model.predict(X_test)
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_test.shape[-1])

    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = mean_squared_error(y_test_flat, y_pred_flat, squared=False)

    print(f'Test MAE: {mae:.4f}')
    print(f'Test RMSE: {rmse:.4f}')

    os.makedirs('models', exist_ok=True)
    model.save('models/model.keras')

    # Save metrics
    with open('models/metrics.txt', 'w') as f:
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

if __name__ == '__main__':
    main()
