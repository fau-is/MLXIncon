import pandas as pd
from scipy.stats import entropy

# Read the text file
with open('../datasets/MI-measure/1T-kbs__MI-measure__max-9-atoms__max-15-elements-with-Consistency-Flag.txt', 'r') as file:
    lines = file.readlines()


# Initialize a list to store the extracted numbers
numbers = []

# Iterate through the lines and extract numbers
for line in lines:
    elements = line.split(",")
    num = float(elements[1].replace(')',''))
    numbers.append(num)

# Create a Pandas DataFrame with the extracted numbers
df = pd.DataFrame({'Numbers': numbers})
#print(df)

# Calculate the maximum value
max_value = df['Numbers'].max()

# Calculate the minimum value
min_value = df['Numbers'].min()

# Calculate the mean value
mean_value = df['Numbers'].mean()

# Calculate the entropy
values = df['Numbers'].value_counts()
print(values)
entropy_value = entropy(values)

# Print the results
print(f"Max Value: {max_value}")
print(f"Min Value: {min_value}")
print(f"Mean Value: {mean_value}")
print(f"Entropy: {entropy_value}")
