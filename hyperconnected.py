import torch
import torch.nn as nn

class HyperLayer(nn.Module):
    def __init__(self, n_inputs, n_outs):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outs = n_outs
        self.layer = nn.Linear(in_features=n_inputs, out_features=n_outs)
        self.feedback = nn.Linear(n_outs, n_outs)
        self.activation = nn.ReLU()
        self.state = torch.zeros((1000, self.n_outs))

    def forward(self, X):
        Z = self.layer(X)
        A = self.activation(Z)

        X = self.layer(X) + self.feedback(A)
        X = self.activation(X)
        return X

model = nn.Sequential(HyperLayer(2, 10), nn.Linear(10, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

X = torch.rand(1000, 2) * 2 - 1
Y = torch.where(X[:, 0] * X[:, 1] >= 0, torch.tensor(0.), torch.tensor(1.))

for _ in range(100):
    pred = model(X)
    loss = criterion(pred, Y)

    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())
