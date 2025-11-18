
import pandas as pd

path = "./results_outs2.csv"
data = pd.read_csv(path, index_col=None)
print("ACC", round(data["acc"].mean(), 3), round(data["acc"].std(), 3))
print("F1", round(data["f1"].mean(), 3), round(data["f1"].std(), 3))