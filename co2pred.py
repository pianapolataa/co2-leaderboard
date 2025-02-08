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


class TextDataset():
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        self.data = self.data.values  # Convert DataFrame to NumPy array before using PyTorch
        self.data = torch.tensor(self.data, dtype=torch.float32)
        
        self.inputs = self.data[:, :-5]
        self.outputs = self.data[:, 1:]
        self.inputs = 2 * (self.inputs - self.inputs.min()) / (self.inputs.max() - self.inputs.min()) - 1
        self.outputs = 2 * (self.outputs - self.outputs.min()) / (self.outputs.max() - self.outputs.min()) - 1

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
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

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

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss =  0
    total_mae = 0  # Variable to accumulate MAE
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            mae = calculate_mae(pred, y)
            total_mae += mae.item()
    test_loss /= num_batches
    total_mae /= num_batches 

    print(f"Test Error: \n Avg loss: {test_loss:.4f} \n Avg MAE: {total_mae:.4f}")
    # return pred.cpu().numpy()

# def gety(dataloader):
#     with torch.no_grad():
#         for X, y in dataloader:
#             y = y.to(device)
#     return y.cpu().numpy()



if __name__ == "__main__":

    # training_data = TextDataset("/Users/sissi/Downloads/pythonstuff/anyskin/anyskin/visualizations/data_2025-02-03_20-39-21.txt")
    # test_data = TextDataset("/Users/sissi/Downloads/pythonstuff/anyskin/anyskin/visualizations/contdata_2025-02-06_11-55-46.txt")

    
    training_data = TextDataset('/Users/sissi/Downloads/table1_train.xlsx')
    test_data = TextDataset('/Users/sissi/Downloads/table1_test.xlsx')

    train_dataloader = DataLoader(training_data, batch_size=10)
    test_dataloader = DataLoader(test_data, batch_size=10)

    epochs = 1000
    for t in range(epochs):
        train(train_dataloader, model, loss_fn, optimizer)
        if (t % 50 == 0): test(test_dataloader, model, loss_fn)

   
    # res = test(test_dataloader, model, loss_fn).T
    # act = gety(test_dataloader).T
    # t = np.zeros(len(res[0]))
    # for i in range(len(res[0])):
    #     t[i] = i
    # plt.plot(t, res[50], label = "predicted", linestyle="-.")
    # plt.plot(t, act[50], label = "actual", linestyle="-")
    # plt.legend()
    # plt.show()



    print("Done!")
    
