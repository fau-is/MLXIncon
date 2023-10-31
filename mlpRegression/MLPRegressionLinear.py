from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import scipy.sparse as sp
import pandas as pd


class RegressionMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the neural network architecture
        super().__init__()

        self.fc1 = nn.Linear(input_size, output_size)

        # self.tanh1 = nn.Tanh()
        #self.hidden_layer_1 = nn.Linear(hidden_size, output_size)

        # self.tanh4 = nn.Tanh()
        # self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Check if the input is sparse, and convert it to a PyTorch Tensor if needed
        if sp.issparse(x):
            x = torch.Tensor(x.toarray())

        y = torch.flatten(x, 1)  # y is one-dimensional
        y = self.fc1(y)
        # y = self.hidden_layer_1(self.tanh1(self.fc1(y)))
        # y = self.fc3(self.tanh4(y))

        return y

    def fit(self, x_train, y_train, epochs=1000, lr=0.01, wd=0.001, lf=torch.nn.MSELoss()):
        # optimizer = torch.optim.NAdam(self.parameters(), lr=lr, weight_decay=wd)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        # Convert sparse input data to PyTorch Tensor if needed
        if sp.issparse(x_train):
            x_train = torch.Tensor(x_train.toarray())

        # Convert target data to PyTorch Tensor
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Training loop
        for epoch in range(epochs):
            y_pred = self(x_train)
            loss = lf(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("epoch " + str(epoch))

filename = "../datasets/1T-kbs__MI-measure__max-3-atoms__max-5-elements.txt"


def read_input_file(filename):
    d = []
    # Open the file for reading
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            content, value = line.split(',', 1)

            # Remove the parentheses and leading/trailing whitespace from the content
            content = content.strip()[2:-1]
            # Convert the value to a float
            value = float(value.strip()[0:-1])

            # Append the parsed data as a tuple to the list
            d.append((content, value))
    return d

seed = 42
torch.manual_seed(seed)


# Sample dataset (replace this with your actual dataset)
# Each row is a set of propositional logic formulas, and the last column is the numerical value.
data = read_input_file(filename)

# Split the dataset into features (X) and labels (y)
X_text = [row[0] for row in data]  # Formulas as text
y = [row[1] for row in data]  # Labels (numerical values)


# Define a custom tokenizer that splits based on spaces
def custom_tokenizer(text):
    return text.split()


# Use CountVectorizer to convert the text data into numerical features
vectorizer = CountVectorizer(binary=True,
                             tokenizer=custom_tokenizer)  # Use binary encoding (1 for presence, 0 for absence)
X = vectorizer.fit_transform(X_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

result = pd.DataFrame({"learning_rate": [],
                       "epochs": [],
                       "weight_decay": [],
                       "MAE_test": [],
                       "MAE_train": []})

lrs = [0.01]
epochs = [1000]
wds = [0.002]

for lr in lrs:
    for epoch in epochs:
        for wd in wds:
            # Create a MLP model
            mlp = RegressionMLP(25, 25, 1)
            # Batch Normalization
            # mlp.BatchNorm2d = torch.nn.BatchNorm2d(4)

            # Train the model on the training data
            mlp.fit(X_train, y_train, epochs=epoch, lf=torch.nn.L1Loss(), lr=lr, wd=wd)

            # Make predictions on the testing data
            y_pred_mlp = mlp(X_test)

            # Make predictions on the training data
            train_y_pred_mlp = mlp(X_train)

            # Evaluate the model
            print("lr: " + str(lr) + " | epochs: " + str(epoch) + " | wd: " + str(wd))
            with torch.no_grad():
                mae_test = mean_absolute_error(y_test, y_pred_mlp)
                print(f"MAE for MLP Reg on test data.: {mae_test}")

                mae_train = mean_absolute_error(y_train, train_y_pred_mlp)
                print(f"MAE for MLP Reg on train data: {mae_train}")

                print()
            # Now, you can use the trained model to predict the value for new sets of propositional logic formulas as text
            # Replace 'new_input_text' with your new input data
            new_input_text = ["!a&&!b b"]  # Example input
            print(f'Example {new_input_text}')
            new_input = vectorizer.transform(new_input_text)
            predicted_value = mlp(new_input)
            print(f"Predicted Value MLP Reg.: {predicted_value[0]}")

            temp = pd.DataFrame({"learning_rate": [lr],
                                 "epochs": [epoch],
                                 "weight_decay": [wd],
                                 "MAE_test": [mae_test],
                                 "MAE_train": [mae_train]})

            result = pd.concat([result, temp])
            print(result)

            print("--- --- --- --- --- ---")

# result.to_csv(r"")
