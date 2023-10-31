from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import pandas as pd


class RegressionMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc1(x)

        return out

    def fit(self, x_train, y_train, epochs=1000, lr=0.01, wd=0.001, criterion=torch.nn.MSELoss()):
        # optimizer = torch.optim.NAdam(self.parameters(), lr=lr, weight_decay=wd)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=wd)

        # Convert target data to PyTorch Tensor
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Training loop
        for epoch in range(epochs):

            optimizer.zero_grad()
            y_pred = self(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            print('epoch {}, loss {}'.format(epoch, loss.item()))

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
X = torch.Tensor(X.toarray())
y = torch.Tensor(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

result = pd.DataFrame({"learning_rate": [],
                       "epochs": [],
                       "weight_decay": [],
                       "MAE_test": [],
                       "MAE_train": []})

lrs = [0.01]
epochs = [100]
wds = [0.002]

for lr in lrs:
    for epoch in epochs:
        for wd in wds:

            # Create model
            mlp = RegressionMLP(25, 25, 1)

            # Train model
            mlp.fit(X_train, y_train, epochs=epoch, lr=lr, wd=wd)

            # Make predictions
            with torch.no_grad():
                y_pred_train = mlp(X_train)
                y_pred_test = mlp(X_test)

            # Evaluate the model
            print("lr: " + str(lr) + " | epochs: " + str(epoch) + " | wd: " + str(wd))
            with torch.no_grad():
                mae_test = mean_absolute_error(y_test, y_pred_test)
                print(f"MAE for MLP Reg on test data.: {mae_test}")

                mae_train = mean_absolute_error(y_train, y_pred_train)
                print(f"MAE for MLP Reg on train data: {mae_train}")

# Now, you can use the trained model to predict the value for new sets of propositional logic formulas as text
# Replace 'new_input_text' with your new input data
new_input_text = ["!a&&!b b"]  # Example input
print(f'Example {new_input_text}')
new_input = vectorizer.transform(new_input_text)
new_input = torch.Tensor(new_input.toarray())
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
