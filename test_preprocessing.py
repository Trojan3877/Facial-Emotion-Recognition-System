import torch
import torchvision.transforms as T

def test_transform_output_shape():
    transform = T.Compose([
        T.Resize((48, 48)),
        T.ToTensor()
    ])

    x = torch.zeros((3, 100, 100))
    y = transform(x)

    assert y.shape == (3, 48, 48), "❌ Transform output shape incorrect"
