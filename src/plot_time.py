import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Size_training_set": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Solver": [31.195, 34.938, 34.741, 234.350, 450.518, 652.805, 2080.243, 5757.570, 8626.182],
    "MLP_flags_data": [81.067, 86.545, 71.892, 70.925, 84.677, 107.225, 81.396, 93.396, 103.850],
    "MLP_flags_model": [247.525, 253.729, 227.228, 236.089, 252.624, 276.675, 267.891, 284.563, 315.974]
})

plt.plot(df["Size_training_set"], df["Solver"], marker = "o", linestyle="--", label="Solver")
plt.plot(df["Size_training_set"], df["MLP_flags_data"], marker = "o", linestyle="--", label="MLP (with flags)")
plt.plot(df["Size_training_set"], df["MLP_flags_model"], marker = "o", linestyle="--", label="MLP (with flags+constraints)")

x_labels = ["3-5", "6-5", "9-5", "3-10", "6-10", "9-10", "3-15", "6-15", "9-15"]
plt.xticks(df["Size_training_set"], x_labels)

y = [2000, 4000, 6000, 8000, 10000]
plt.yticks(y)

plt.legend(loc="upper left")

plt.xlabel("Data set size")
plt.ylabel("Total runtime")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'data_sets_total_runtime.pdf', dpi=100, bbox_inches="tight")
plt.close(fig1)