from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
import numpy as np


filename ="../datasets/1T-kbs__AT-measure__max-4-atoms__max-5-elements_with-Consistency-and-UB-Flag.txt"

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
model_linear = LinearRegression()
model_logistic = LogisticRegression()


# Train the model on the training data
model_linear.fit(X_train, y_train)
model_logistic.fit(X_train, y_train)



# Make predictions on the testing data
y_pred_linear = model_linear.predict(X_test)
y_pred_logistic = model_logistic.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred_linear)
print(f"Mean Absolute Error Linear Reg.: {mae}")

mae2 = mean_absolute_error(y_test, y_pred_logistic)
print(f"Mean Absolute Error Logistic Reg.: {mae2}")

print()
# Now, you can use the trained model to predict the value for new sets of propositional logic formulas as text
# Replace 'new_input_text' with your new input data
new_input_text = ["!a&&b !b"]  # Example input
print(f'Example {new_input_text}')
new_input = vectorizer.transform(new_input_text)
predicted_value = model_linear.predict(new_input)
print(f"Predicted Value Linear Reg.: {predicted_value[0]}")

predicted_value = model_logistic.predict(new_input)
print(f"Predicted Value Logistic Reg.: {predicted_value[0]}")