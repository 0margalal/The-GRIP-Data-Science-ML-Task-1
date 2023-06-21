# Omar Galal Hassan Marghany

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# The main objective of this Python code is to build a Supervised ML model specifically
# #Linear Regression to predict the percentage of a student score based on the no. of hours he/she studies.

# I Imported the libraries I may need like Sklearn, Pandas, and Numpy.

Data = pd.read_csv(r"C:\Users\Omar Galal Hassan\Desktop\THE GRIP\Students Data.csv")
# Now I opened the given CSV file that has two variables (Score , No. of Hours)

print("\nThe Data :\n", Data.to_string(), '\n',"\nTop 5 Rows :\n", Data.head(), '\n',"\nData's Info :\n", Data.info(), '\n',   "\nDescription about the Data : \n", Data.describe(), '\n')
# First let me show you the data and its description

plt.style.use('dark_background')
plt.scatter(Data.Hours , Data.Scores)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Students Data')
plt.show()

x = Data[['Hours']].to_numpy()
y = Data['Scores'].to_numpy()
# I created 2 variables to transfer the columns so that I can use them

X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=42)
# I created the Training and Testing variables and I chose the test size 0.2 so that I could test 20% of the Data
# And Random State = 42 is the most specific number that could make the data random as possible

model = LinearRegression()
# Creating variable model to create my Liner Regression Model

model.fit(X_train1, y_train1)
# Giving the Variable Model the Training Data

y_pred1 = model.predict(X_test1)
# Getting the results and Prediction

Evaluation = pd.DataFrame({'Actual Values':y_test1,'Predicted Values':y_pred1})
print('\n',Evaluation,'\n')
# A DataFrame to compare between the Actual and Preidicted Values

mape = mean_absolute_percentage_error(y_test1, y_pred1)
print('Mean absolute percentage error (MAPE) in percentage : {:.1f}%'.format(mape*100))

# Printing the mean absolute percentage error that show the difference between actual and predicted data
# It should be least as possible (Maximum 10%).

print('R-squared score (training): {:.3f}'.format(model.score(X_train1, y_train1)))
print('R-squared score (test): {:.3f}'.format(model.score(X_test1, y_test1)))
# The R squared is a test that presents how well the model fits the dependent
# variables (close to 1 as much as possible or even more).

# Linear Regression:
# - Mean Absolute Percentage Error (MAPE): 0.1%
# - R-squared score (training): 0.949
# - R-squared score (test): 0.968

plt.style.use('dark_background')
plt.scatter(x,y)
plt.plot(x,model.predict(x),color='red')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Values and Linear Model')
plt.show()
# A Scatter plot that shows the Actual Values of the Data and the Linear Model Predictions

given_test = 9.25
# I will give the model the given Independent Variable 'Hours' as asked for.

given_test = np.array(given_test).reshape(-1, 1)
# Reshaping the Independent Variable 'Hours'

Output = model.predict(given_test)
# Predicting  the Dependent Variable 'Scores'

print("\nPredicted Score for 9.25 hours per day :", Output)
