import pandas as pd
df = pd.read_csv("C:/Users/user/Downloads/spaceship-titanic/test.csv")
df.info()
df.dtypes
print(df.head())
cols = df.columns
obj = 'Name'
if obj in cols:
    print('Dataset has column Name')
obj = 'Drowned'
if obj not in cols:
    print('Dataset hasn’t column Drowned')