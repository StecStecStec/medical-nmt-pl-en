import pandas as pd

df = pd.read_csv("data_testing.csv", header=None, names=["polish", "english"])

df_short = df.truncate(before=0, after=9999)

df_short.to_csv("data_testing_short.csv", index=False, header=False)