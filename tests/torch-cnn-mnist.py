import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .args import parse, read_sets_compat

TAG="test-cnn-mnist"
NUM_CORES=0
NUM_EPOCHS=200 # 5
CHANNELS=32
BATCH_SIZE=128  # 64
LEARNING_RATE=0.005
# print(multiprocessing.get_all_start_methods())
# multiprocessing.set_start_method('spawn')
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


args = parse(TAG)
mlop, settings = read_sets_compat(args, TAG)



run = mlop.init(dir=".mlop/", project=TAG, settings=settings)

# define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(CHANNELS * 14 * 14, 10)  #  28x28 MNIST images

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc1(x)
        return x

# load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.MNIST(
    root="./tests", train=True, transform=transform, download=True
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_CORES)

# initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    model.train()
    return avg_loss

# training loop
def train_model(model, train_loader, criterion, optimizer, epochs=NUM_EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # log batch loss and example images with predictions
            if batch_idx % 100 == 0:
                print(f"Batch [{batch_idx}/{len(train_loader)}], Batch Loss: {loss.item():.4f}")
                run.log({"loss/batch": loss.item(),})

        avg_loss = total_loss / len(train_loader)
        validation_loss = evaluate_model(model, train_loader, criterion)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_loss:.4f}", f"Validation Loss: {validation_loss:.4f}")
        run.log({"epoch": epoch + 1, "loss/train": avg_loss, "loss/val": validation_loss})


# train the model
train_model(model, train_loader, criterion, optimizer, epochs=NUM_EPOCHS)

run.finish()
