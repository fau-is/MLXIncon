import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Size_training_set": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Ridge_flags": [0.570, 0.538, 0.528, 0.514, 0.501, 0.492, 0.485, 0.478, 0.472],
    "Lasso_flags": [0.599, 0.569, 0.559, 0.544, 0.525, 0.515, 0.507, 0.504, 0.501],
    "MLP_flags": [0.564, 0.527, 0.513, 0.499, 0.483, 0.474, 0.466, 0.459, 0.450],
    "MLP_flags_constraints": [0.568, 0.526, 0.511, 0.502, 0.485, 0.470, 0.466, 0.459, 0.453]
})

plt.plot(df["Size_training_set"], df["Ridge_flags"], marker = "o", linestyle="--", label="Ridge (with flags)")
plt.plot(df["Size_training_set"], df["Lasso_flags"], marker = "o", linestyle="--", label="Lasso (with flags)")
plt.plot(df["Size_training_set"], df["MLP_flags"], marker = "o", linestyle="--", label="MLP (with flags)")
plt.plot(df["Size_training_set"], df["MLP_flags_constraints"], color="grey", marker = "o", linestyle="--", label="MLP (with flags+constraints)")

x_labels = ["1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k", "9k"]
plt.xticks(df["Size_training_set"], x_labels)

y = [0.425, 0.450, 0.475, 0.500, 0.525, 0.550, 0.575, 0.6]
plt.yticks(y)
plt.ylim(0.42, 0.61)

plt.legend(loc="lower left")

plt.xlabel("Training sample size")
plt.ylabel("Test MAE")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'sample_sizes_1k-9k_cv.pdf', dpi=100, bbox_inches="tight")
plt.close(fig1)