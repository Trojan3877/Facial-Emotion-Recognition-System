def test_model_loads():
    from src.model import load_model
    model = load_model()
    assert model is not None

def test_inference_output_shape():
    import numpy as np
    from src.model import predict
    dummy_input = np.zeros((1, 48, 48, 1))
    output = predict(dummy_input)
    assert output.shape[-1] == 7
