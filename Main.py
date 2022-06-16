import pandas as pd
import GradientDescent
from Plotter import *

'''King County House Prices'''
file = open('kc_house_data.csv', 'r', newline='')
data = pd.DataFrame(pd.read_csv(file))

# Living Space vs Price
x = data['sqft_living'] / 1000
y = data['price'] / 1000
GradientDescent.L = 0.1
GradientDescent.max_epochs = 10000
GradientDescent.threshold = 1e-7
plotter(x, y, x_label="Living Space (1,000's sqft)", y_label="Price (1,000's)", title="Living Space vs Price")

# Lot size vs Price
x = data['sqft_lot'] / 10000
y = data['price'] / 1000
GradientDescent.L = 0.03
GradientDescent.threshold = 1e-7
plotter(x, y, x_label="Lot (10,000's sqft)", y_label="Price (1,000's)", title="Lot Size vs Price")
file.close()

# Bedrooms vs Price
x = data['bedrooms']
y = data['price'] / 1000
GradientDescent.L = 0.001
GradientDescent.threshold = 1e-7
plotter(x, y, x_label="Number of Bedrooms", y_label="Price (1,000's)", title="Bedroom Count vs Price")
file.close()
