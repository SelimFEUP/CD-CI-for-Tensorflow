import numpy as np
from src.model import build_model
import tensorflow as tf

def test_model_build():
    input_shape = (24, 5)     # 24 time steps, 5 features
    output_shape = (24, 5)    # 24 time steps, 5 features predicted
    model = build_model(input_shape, output_shape)
    
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, *output_shape)

def test_model_training_step():
    input_shape = (24, 5)
    output_shape = (24, 5)
    model = build_model(input_shape, output_shape)

    # Create small dummy dataset
    X_dummy = np.random.rand(10, 24, 5).astype(np.float32)
    y_dummy = np.random.rand(10, 24, 5).astype(np.float32)

    # Train 1 epoch
    history = model.fit(X_dummy, y_dummy, epochs=1, batch_size=2, verbose=0)
    
    # Check if training loss exists
    assert 'loss' in history.history
