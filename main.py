import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("processed_titanic.csv")
X = df.drop(['Spa'], axis=1)
y = df['Spa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_test = linear_model.predict(X_test)

print("Оценка регресии")
MSE = mean_squared_error(y_test, y_pred_test)
print(MSE)
RMSE = root_mean_squared_error(y_test, y_pred_test)
print(RMSE)
MAE = mean_absolute_error(y_test, y_pred_test)
print(MAE)

X1 = df.drop(['Transported'], axis=1)
y1 = df['Transported']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.4, random_state=42)
X1_test, X1_val, y1_test, y1_val = train_test_split(X1_test, y1_test, test_size=0.4, random_state=42)

scaler = StandardScaler()
X1_train_scaled = scaler.fit_transform(X1_train)
X1_test_scaled = scaler.transform(X1_test)


logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X1_train_scaled, y1_train)
y1_pred_test = logreg_model.predict(X1_test_scaled)

accuracy = accuracy_score(y1_test, y1_pred_test)
print("Точность: " , accuracy)


cm = confusion_matrix(y1_test, y1_pred_test)

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()



