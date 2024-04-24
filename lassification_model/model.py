# model 384 -> 12
from torch import nn


# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.65)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.65)

        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # out = self.fc22(out)
        # out = self.relu(out)
        # out = self.dropout22(out)

        out = self.fc3(out)

        return out

    def embed(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        # out = self.fc22(out)
        # out = self.relu(out)
        return out
