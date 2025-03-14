import pandas as pd
import time
import torch
from torch import nn

from data import MNISTData


class TrainConfig:
    """
    Small class for holding information related to training parameters.
    """
    def __init__(self, epochs: int = 10, learning_rate: float = 0.001, optimizer: str = None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.loss_fn = nn.CrossEntropyLoss()

class BaseNetwork(nn.Module):
    """
    Template for network to be used when reloading pretrained weights.
    """
    def __init__(self, hidden_layer_sizes: list = [84]):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []

        for i, size in enumerate(hidden_layer_sizes):
            if i == 0:
                layers.append(nn.Linear(28*28, size))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden_layer_sizes[i-1], size))
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], 10))

        self.linear_relu_stack = nn.Sequential(*layers)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the neural network.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetwork(BaseNetwork):
    """
    Main network for training, incorporating data and optimizer.
    """
    def __init__(self, hidden_layer_sizes: list = [84], train_config: TrainConfig = None, data: MNISTData = None):
        super().__init__(hidden_layer_sizes)

        self.train_config = train_config
        self.data = data
        self.hidden_layer_sizes = hidden_layer_sizes

        self.optimizer =  torch.optim.Adam(self.parameters(), lr=self.train_config.learning_rate) if self.train_config.optimizer_name == 'adam' else torch.optim.SGD(self.parameters(), lr=self.train_config.learning_rate)

    def train_loop(self):
        """
        Perform one training loop on the data.
        """
        self.train()

        running_loss = 0.0
        for batch, (X, y) in enumerate(self.data.train_loader):
            # X = X.to(device)
            # y = y.to(device)

            pred = self(X)
            loss = self.train_config.loss_fn(pred, y)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()

        return running_loss / len(self.data.train_loader.dataset)

    def validation_loop(self):
        """
        Perform one validation loop on the data.
        """
        self.eval()

        running_loss = 0.0
        for batch, (X, y) in enumerate(self.data.val_loader):
            # X = X.to(device)
            # y = y.to(device)

            with torch.no_grad():
                pred = self(X)
                loss = self.train_config.loss_fn(pred, y)
                running_loss += loss.item()

        return running_loss / len(self.data.val_loader.dataset)

    def full_train(self):
        """
        Train the model on the training set and validate on the validation set.
        """
        train_losses = []
        val_losses = []

        for epoch in range(self.train_config.epochs):
            train_losses += [self.train_loop()]
            val_losses += [self.validation_loop()]

        loss_df = pd.concat([
            pd.DataFrame({
                'loss': train_losses,
                'type': len(train_losses) * ['train'],
            }).reset_index(),
            pd.DataFrame({
                'loss': val_losses,
                'type': len(val_losses) * ['val']
            }).reset_index()
        ])

        return loss_df

    def get_accuracy(self):
        """
        Get the accuracy of the model on the test set.
        """
        with torch.no_grad():
            num_correct = 0
            for (X, y) in self.data.test_loader:
                pred = nn.Softmax(dim=1)(self(X))
                num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            accuracy = num_correct / len(self.data.test_loader.dataset)
            return accuracy

    def run_training(self):
        """
        Run the full training loop and record performance for later analysis.
        """
        now = time.time()
        loss_df = self.full_train()
        runtime = time.time() - now
        accuracy = self.get_accuracy()
        return loss_df, pd.DataFrame({
            'accuracy': [accuracy],
            'runtime': [runtime],
            'epochs': [self.train_config.epochs],
            'learning_rate': [self.train_config.learning_rate],
            'hidden_layer_sizes': [self.hidden_layer_sizes],
            'optimizer': [self.train_config.optimizer_name],
            'batch_size': [self.data.batch_size],
            'normalize': [self.data.normalize],
            'val_split': [self.data.val_split],
        })
