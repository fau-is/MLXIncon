import torch.nn as nn
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_val_loss = np.inf

    def __call__(self, val_loss):
        if (val_loss - self.best_val_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.counter = 0
            self.best_val_loss = val_loss


class RegressionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.loss_val = np.inf

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)

        return out

    def fit(self, x_train, y_train, x_val, y_val,
            patients=10, epochs=100, learning_rate=0.001, weight_decay=0.001):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Convert target data to PyTorch Tensor
        y_train = y_train.clone().detach()
        y_train = y_train.reshape(-1, 1).float()

        early_stopping = EarlyStopping(tolerance=patients, min_delta=0)
        loss_function = torch.nn.L1Loss()

        # Training loop
        for epoch in range(epochs):

            optimizer.zero_grad()
            y_train_pred = self(x_train)
            loss_train = loss_function(y_train_pred, y_train)
            loss_train.backward()
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss_train.item()))

            # Validation
            with torch.no_grad():
                y_val_pred = self(x_val)
                loss_val = loss_function(y_val_pred, y_val)

            # Early stopping
            early_stopping(loss_val)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break

        self.loss_val = loss_val


class RegressionMLPNS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, target):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.loss_val = np.inf
        self.target = target

    def forward(self, x, x_flags):
        out = self.fc1(x)
        out = self.tanh1(out)
        out = self.fc2(out)
        out = self.tanh2(out)
        out = self.fc3(out)

        return out

    def custom_loss_mi(self, y_train_pred, y_train, x_train_flags):
        """Calculates the custom loss for the mi measure"""

        # Prediction loss
        l_n = torch.nn.L1Loss()
        l_n = l_n(y_train_pred, y_train)

        # Con flag loss
        l_con = x_train_flags[:, 0]

        return l_n + ((sum(l_con) / len(l_con)) * l_n)

    def custom_loss_at(self, y_train_pred, y_train, x_train_flags):
        """Calculates the custom loss for the at measure"""

        num_ub_flags = x_train_flags.shape[1] - 1
        ub_flags = []
        ub_flag_sum = 0

        # Prediction loss
        l_n = torch.nn.L1Loss()
        l_n = l_n(y_train_pred, y_train)

        # Con flag loss
        l_con = x_train_flags[:, 0]

        # UB flag loss
        for i in range(0, num_ub_flags):
            ub_flags.append([i for i in np.array(x_train_flags[:, 1 + i])])
            ub_flag_sum += sum(ub_flags[i]) / len(ub_flags[i]) * l_n

        return l_n + (sum(l_con) / len(l_con)) * l_n + ub_flag_sum

    def fit(self, x_train, x_train_flags, y_train, x_val, x_val_flags, y_val,
            patients=10, epochs=1000, learning_rate=0.01, weight_decay=0.001):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Convert target data to PyTorch Tensor
        y_train = y_train.clone().detach()
        y_train = y_train.reshape(-1, 1).float()

        early_stopping = EarlyStopping(tolerance=patients, min_delta=0)

        # Training loop
        for epoch in range(epochs):

            optimizer.zero_grad()
            y_train_pred = self(x_train, x_train_flags)
            if self.target == "AT":
                loss_train = self.custom_loss_at(y_train_pred, y_train, x_train_flags)
            else:
                loss_train = self.custom_loss_mi(y_train_pred, y_train, x_train_flags)

            loss_train.backward()
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss_train.item()))

            # Validation
            with torch.no_grad():
                y_val_pred = self(x_val, x_val_flags)
                loss_val = self.custom_loss_at(y_val_pred, y_val, x_val_flags)

            # Early stopping
            early_stopping(loss_val)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break

        self.loss_val = loss_val
