import pandas as pd
df = pd.read_csv("C:/Users/user/PycharmProjects/AILabs/Lab1/spaceship-titanic/test.csv")
df.info()
df.dtypes
print(df.head(100))
cols = df.columns

nan_matrix = df.isnull()
print(nan_matrix.head(100))
print(nan_matrix.sum())

nan_matrix = df.isnull()

cabin_mode = df['Cabin'].mode()[0]
print(cabin_mode)
df['Cabin'].fillna(cabin_mode)
nan_matrix = df.isnull()