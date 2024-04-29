from torch import nn
class ClassifierModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    self.fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax()
    )
  def forward(self, x):
    x = self.conv_layers(x)
    x = self.fc_layers(x)
    return x