import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Size_training_set": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Ridge_flags": [0.690, 0.668, 0.640, 0.618, 0.604, 0.590, 0.584, 0.581, 0.579],
    "Lasso_flags": [0.728, 0.693, 0.628, 0.625, 0.608, 0.607, 0.596, 0.603, 0.609],
    "MLP_flags": [0.726, 0.637, 0.607, 0.601, 0.583, 0.573, 0.605, 0.568, 0.555],
    "MLP_flags_constraints": [0.757, 0.637, 0.621, 0.595, 0.603, 0.573, 0.597, 0.578, 0.558]
})

plt.plot(df["Size_training_set"], df["Ridge_flags"], marker = "o", linestyle="--", label="Ridge (with flags)")
plt.plot(df["Size_training_set"], df["Lasso_flags"], marker = "o", linestyle="--", label="Lasso (with flags)")
plt.plot(df["Size_training_set"], df["MLP_flags"], marker = "o", linestyle="--", label="MLP (with flags)")
plt.plot(df["Size_training_set"], df["MLP_flags_constraints"], color="grey", marker = "o", linestyle="--", label="MLP (with flags+constraints)")

x_labels = ["1k", "2k", "3k", "4k", "5k", "6k", "7k", "8k", "9k"]
plt.xticks(df["Size_training_set"], x_labels)

y = [.5, .55, .6, .65, .7, .75]
plt.yticks(y)

plt.legend(loc="lower left")

plt.xlabel("Training sample size")
plt.ylabel("Test MAE")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'sample_sizes_1k-9k_cv.pdf', dpi=100, bbox_inches="tight")
plt.close(fig1)