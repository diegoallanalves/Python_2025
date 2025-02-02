# Import and display first five rows of advertising dataset

import pandas as pd
import matplotlib.pyplot as plt

advert = pd.read_csv('C:\\Users\\diego\\Desktop\\data_set\\advertising.csv')

print(advert.head())

import statsmodels.formula.api as smf

# Initialise and fit linear regression model using `statsmodels`
model = smf.ols('sales ~ TV', data=advert)
model = model.fit()

# Predict values
sales_pred = model.predict()

# Predict values
sales_pred = model.predict()

# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(advert['TV'], advert['sales'], 'o')           # scatter plot showing actual data
plt.plot(advert['TV'], sales_pred, 'r', linewidth=2)   # regression line
plt.xlabel('TV Advertising Costs')
plt.ylabel('sales')
plt.title('TV vs sales')

plt.show()

new_X = 500
print(model.predict({"TV": new_X}))

from sklearn.linear_model import LinearRegression

# Build linear regression model using TV and Radio as predictors
# Split data into predictors X and output Y
predictors = ['TV', 'radio']
X = advert[predictors]
y = advert['sales']

# Initialise and fit model
lm = LinearRegression()
model = lm.fit(X, y)

print(f'alpha = {model.intercept_}')
print(f'betas = {model.coef_}')

print(model.predict(X))

new_X = [[300, 200]]
print(model.predict(new_X))






