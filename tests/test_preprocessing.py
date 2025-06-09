import numpy as np
from src.data_utils import load_data, create_sequences

def test_load_data():
    data = load_data('data/transformed_data.csv')
    assert isinstance(data, np.ndarray)
    assert data.shape[1] > 0

def test_create_sequences():
    dummy_data = np.random.rand(100, 5)
    X, y = create_sequences(dummy_data, 10, 5)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 10
    assert y.shape[1] == 5
