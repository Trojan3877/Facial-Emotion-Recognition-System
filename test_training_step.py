import torch
from src.model import EmotionCNN

def test_training_step_runs():
    model = EmotionCNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(4, 1, 48, 48)  # grayscale (1 channel)
    y = torch.randint(0, 7, (4,))

    output = model(x)
    loss = criterion(output, y)

    loss.backward()
    optimizer.step()

    assert loss.item() > 0, "❌ Training loss should be > 0"
