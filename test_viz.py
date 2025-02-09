import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd


device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

def calculate_mae(predictions, targets):
    return torch.mean(torch.abs(predictions - targets))

maxi = 681
mini = 2.4

class TextDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path, header = None)
        self.data = self.data.values  # Convert DataFrame to NumPy array before using PyTorch
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        
        self.inputs = self.data[:, :-5]
        self.outputs = self.data[:, -5:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(48, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.007)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)
        # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, record):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss =  0
    total_mae = 0  # Variable to accumulate MAE
    all_preds = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            mae = calculate_mae(pred, y)
            total_mae += mae.item()

            all_preds.append(pred.cpu().numpy())

    test_loss /= num_batches
    total_mae /= num_batches 

    print(f"Test Error: \n Avg loss: {test_loss:.4f} \n Avg MAE: {total_mae:.4f}")
    if (record == True):
        all_preds = np.concatenate(all_preds, axis=0)  
        all_preds = [x * (maxi - mini) + mini for x in all_preds]
        all_preds = np.array(all_preds, dtype = np.float32)
        return all_preds

def gety(dataloader):
    all_y = []
    with torch.no_grad():
        for _, y in dataloader:
            all_y.append(y.cpu().numpy())
    all_y = np.concatenate(all_y, axis=0)
    all_y = all_y * (maxi - mini) + mini
    return all_y


if __name__ == "__main__":
    training_data = TextDataset('table1_train.xlsx')
    test_data = TextDataset('table1_test.xlsx')
    pred_data = TextDataset('table1_copy.xlsx')

    train_dataloader = DataLoader(training_data, batch_size=10, drop_last = False)
    test_dataloader = DataLoader(test_data, batch_size=11, drop_last = False)
    pred_dataloader = DataLoader(pred_data, batch_size=51, drop_last=False)

    print(len(test_dataloader.dataset))

    epochs = 2000
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        if (t % 50 == 0): test(test_dataloader, model, loss_fn, False)

    
    pred = test(test_dataloader, model, loss_fn, True)
    actual = gety(test_dataloader)


    t = np.zeros(len(pred[2]))
    for i in range(len(pred[2])):
        t[i] = i
    plt.ylim(45, 80)
    plt.plot(t, pred[0], label = "predicted", linestyle="-.")
    plt.plot(t, actual[0], label = "actual", linestyle="-")
    plt.legend()
    plt.show()




    print("Done!")
    
