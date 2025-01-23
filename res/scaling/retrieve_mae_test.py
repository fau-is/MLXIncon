import csv
import os

if __name__ == "__main__":

    values = []
    file_names = os.listdir()[0:-1]

    for file_name in file_names:

        with open(file_name, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                if "mae" in str(row).replace(",", "+").split(";")[18]:
                    values.append(str(row).replace(",", "+").split(";")[18])  # MAE run10
                else:
                    values.append(str(round(float(str(row).replace(",", "+").split(";")[18]),3)) + " (+/-" + str(round(float(str(row).replace(",", "+").split(";")[19]),3)) + ")")  # MAE run10

x = 0
for i, value in enumerate(values):
    if "mae" in value:
        print(file_names[x])
        x = x + 1
    print(value)