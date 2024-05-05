import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Size_training_set": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Solver": [24.0080, 18.8490, 15.8490, 249.0980, 460.8910, 508.8330, 1906.2420, 6789.9960, 12801.9170],
    "MLP_flags_data": [1369.9415, 1476.7554, 815.9468, 1070.5932, 1068.8559, 835.4876, 2051.5293, 1105.4310, 788.5607],
    "MLP_flags_model": [6237.0857, 6094.4333, 2814.3300, 4527.8936, 4831.1098, 3646.6979, 8145.6960, 4704.4828, 3445.8381]
})

plt.plot(df["Size_training_set"], df["Solver"], marker = "o", linestyle="--", label="Solver")
plt.plot(df["Size_training_set"], df["MLP_flags_data"], marker = "o", linestyle="--", label="MLP (with flags)")
plt.plot(df["Size_training_set"], df["MLP_flags_model"], marker = "o", linestyle="--", label="MLP (with flags+constraints)")

x_labels = ["3-5", "6-5", "9-5", "3-10", "6-10", "9-10", "3-15", "6-15", "9-15"]
plt.xticks(df["Size_training_set"], x_labels)

y = [2500, 5000, 7500, 10000, 12500, 15000]
plt.yticks(y)

plt.legend(loc="upper left")

plt.xlabel("Data set size")
plt.ylabel("Total runtime")
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(f'data_sets_total_runtime.pdf', dpi=100, bbox_inches="tight")
plt.close(fig1)