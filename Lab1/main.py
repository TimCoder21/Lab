import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.read_csv("C:/Users/user/PycharmProjects/AILabs/Lab1/spaceship-titanic/train.csv")
df.info()
df.dtypes
print(df.head(100))
cols = df.columns

nan_matrix = df.isnull()
print(nan_matrix.head(100))
print(nan_matrix.sum())

nan_matrix = df.isnull()

rood_median = df['RoomService'].median()
age_median = df['Age'].median()
food_median = df['FoodCourt'].median()
shopping_median = df['ShoppingMall'].median()
vip_mean = df['VIP'].mean()
spa_mean = df['Spa'].mean()
vrdeck_mean = df['VRDeck'].mean()
cabin_mode = df['Cabin'].mode()[0]
planet_mode = df['HomePlanet'].mode()[0]
sleep_mode = df['CryoSleep'].mode()[0]
destination_mode = df['Destination'].mode()[0]
name_mode = df['Name'].mode()[0]

df.fillna({'Cabin': cabin_mode}, inplace=True)
df.fillna({'HomePlanet': planet_mode}, inplace=True)
df.fillna({'CryoSleep': sleep_mode}, inplace=True)
df.fillna({'Destination': destination_mode}, inplace=True)
df.fillna({'Name': name_mode}, inplace=True)
df.fillna({'RoomService': rood_median}, inplace=True)
df.fillna({'Age':age_median}, inplace=True)
df.fillna({'ShoppingMall': shopping_median}, inplace=True)
df.fillna({'FoodCourt':food_median}, inplace=True)
df.fillna({'VIP':vip_mean}, inplace=True)
df.fillna({'Spa':spa_mean}, inplace=True)
df.fillna({'VRDeck':vrdeck_mean}, inplace=True)

nan_matrix = df.isnull()
print(nan_matrix.sum())

scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])
scaler = MinMaxScaler()
df['RoomService'] = scaler.fit_transform(df[['RoomService']])
scaler = MinMaxScaler()
df['FoodCourt'] = scaler.fit_transform(df[['FoodCourt']])
scaler = StandardScaler()
df['ShoppingMall'] = scaler.fit_transform(df[['ShoppingMall']])
scaler = StandardScaler()
df['Spa'] = scaler.fit_transform(df[['Spa']])
scaler = StandardScaler()
df['VRDeck'] = scaler.fit_transform(df[['VRDeck']])

df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)
df = pd.get_dummies(df, columns=['CryoSleep'], drop_first=True)
df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)
df = pd.get_dummies(df, columns=['Destination'], drop_first=True)
df = pd.get_dummies(df, columns=['VIP'], drop_first=True)
df = pd.get_dummies(df, columns=['Name'], drop_first=True)

df.to_csv("processed_titanic.csv", index=False)
