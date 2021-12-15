import torch.nn as nn
import torch


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        super(BiLSTM, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # nodes in each layer, hidden layer size
        self.hidden_dim = hidden_dim

        # number of layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, bidirectional=True, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        # we need to multiply the layer dimension with 2 because we need to do forward and backward
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        # x.size(0) = number of example
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(self.device)
        h0.requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(self.device)
        c0.requires_grad_()

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))

        # take only the last hidden state to sent into the linear layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out