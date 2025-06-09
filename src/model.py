import tensorflow as tf

def build_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # (input_steps, num_features)

        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(output_shape[0] * output_shape[1]),

        tf.keras.layers.Reshape(output_shape)  # (output_steps, num_features)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae'])
    return model
