import math
import platform
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from .args import parse, read_sets_compat

TAG = "test-polynomial"
if platform.system() == "Linux" or platform.machine() == "x86_64":
    config = {"epochs": 10_000, "learning_rate": 0.01}  # for actions
else:
    config = {"epochs": 500_000, "learning_rate": 0.0001}

args = parse(TAG)
mlop, settings = read_sets_compat(args, TAG)



run = mlop.init(dir=".mlop/", project=TAG, settings=settings,)

# generate synthetic data for polynomial regression
x = torch.linspace(-1, 1, 100).view(-1, 1)
y = 2 * x**3 + 3 * x**2 - x + torch.randn(x.size()) * 0.1  # polynomial with noise

class PolynomialRegression(nn.Module):
    def __init__(self):
        super(PolynomialRegression, self).__init__()
        self.fc = nn.Linear(4, 1)  # 4 coefficients for cubic regression

    def forward(self, x):
        x_poly = torch.cat([x**i for i in range(4)], dim=1)  # [1, x, x^2, x^3]
        return self.fc(x_poly)

# initialize the model, loss function, and optimizer
model = PolynomialRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])  # may hang

def train_model(model, x, y, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        # forward pass
        predictions = model(x)
        loss = criterion(predictions, y)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        run.log(
            {
                "epoch": epoch + 1,
                "loss": loss.item(),
                "val/loss": loss.item(),
                "train/loss": loss.item(),
                "info/time": time.time(),
            }
        )
        run.log({"val/random": random.random()})
        run.log({"val/sine": math.sin(epoch)})

        if epoch % 1000 == 0:
            print(f"Epoch {epoch + 1} / {config['epochs']}")

train_model(model, x, y, criterion, optimizer, config["epochs"])
