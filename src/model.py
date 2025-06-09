import tensorflow as tf

def build_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(output_shape[0] * output_shape[1]),
        tf.keras.layers.Reshape(output_shape)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
