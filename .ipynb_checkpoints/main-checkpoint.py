import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/user/PycharmProjects/AILabs/Lab2/processed_titanic.csv")
X = df.drop(['PassengerId'], axis=1)
y = df['PassengerId']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)


linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_test = linear_model.predict(X_test)

MSE = mean_squared_error(y_test, y_pred_test)