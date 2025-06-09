import numpy as np
from data_utils import load_data, preprocess_data, create_sequences, split_data
from model import build_model
import tensorflow as tf
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
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)
    loss, mae = model.evaluate(X_test, y_test)

    print(f'Test MAE: {mae:.4f}')

    os.makedirs('models', exist_ok=True)
    model.save('models/traffic_model')

if __name__ == '__main__':
    main()
