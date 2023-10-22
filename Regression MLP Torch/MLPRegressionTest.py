from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np


class RegressionMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 3 linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if sp.issparse(x):
            x = torch.Tensor(x.toarray())

        y = torch.flatten(x,1) #y is one-dimensional

        y = nn.GELU()(self.fc1(y))
        y = nn.GELU()(self.fc2(y))
        y = self.fc3(y)

        return y

    def fit(self, x_train, y_train,epochs=1000, learning_rate=0.01, weight_decay=0.001, loss_function=torch.nn.MSELoss()):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        if sp.issparse(x_train):
            x_train = torch.Tensor(x_train.toarray())
        y_train = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(epochs):
            y_pred = self(x_train)
            loss = loss_function(y_pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("epoch " + str(epoch) +": done")



filename ="../datasets/10T-kbs__MI-measure__max-3-atoms__max-5-elements.txt"


def read_input_file(filename):
    d=[]
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



# Sample dataset (replace this with your actual dataset)
# Each row is a set of propositional logic formulas, and the last column is the numerical value.
data = read_input_file(filename)

# Split the dataset into features (X) and labels (y)
X_text = [row[0] for row in data]  # Formulas as text
y = [row[1] for row in data]       # Labels (numerical values)

# Define a custom tokenizer that splits based on spaces
def custom_tokenizer(text):
    return text.split()

# Use CountVectorizer to convert the text data into numerical features
vectorizer = CountVectorizer(binary=True,tokenizer=custom_tokenizer)  # Use binary encoding (1 for presence, 0 for absence)
X = vectorizer.fit_transform(X_text)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a Linear Regression model
mlp = RegressionMLP(25,32,1)

# Train the model on the training data
mlp.fit(X_train, y_train)


# Make predictions on the testing data
y_pred_mlp = mlp(X_test)

# Evaluate the model
with torch.no_grad():
    mae = mean_absolute_error(y_test, y_pred_mlp)
    print(f"Mean Absolute Error Linear Reg.: {mae}")


    print()
# Now, you can use the trained model to predict the value for new sets of propositional logic formulas as text
# Replace 'new_input_text' with your new input data
    new_input_text = ["!a&&!b b"]  # Example input
    print(f'Example {new_input_text}')
    new_input = vectorizer.transform(new_input_text)
    predicted_value = mlp(new_input)
    print(f"Predicted Value Linear Reg.: {predicted_value[0]}")
