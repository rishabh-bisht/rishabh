import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported")
print(data.head(10))


data.plot(x="Hours", y="Scores", style="*")
plt.title("Hours vs Scores")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

x = data.iloc[:,:-1].values
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print("training complete")
line = regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


from sklearn.linear_model import LinearRegression
lireg = LinearRegression()
lireg.fit(x_train,y_train)

print(x_test)
y_pred = lireg.predict(x_test)
df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(df)

hours = 9.25
pred = lireg.predict(np.array([9.25]).reshape(1, 1))
print("No. of hours = {}".format(hours))
print("Predicted scores = {}".format(pred[0]))

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
