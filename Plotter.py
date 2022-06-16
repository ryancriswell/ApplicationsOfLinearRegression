import matplotlib.pyplot as plt
from PartitionPie import *
from GradientDescent import weights, biases, costs


def plotter(x, y, x_label, y_label, title):
    """LEAST SQUARES REGRESSION"""
    # get our slope and intercept
    m, b = ols(x, y)
    print(f'OLS Function: \n y = {m}x + {b}')
    # plot our points
    plt.scatter(x, y, s=3, color='#1AA7EC')
    # add our Simple Linear Regression model
    plt.plot(x, (m * x + b), color='#CC5801', label='OLS')

    """GRADIENT DESCENT"""
    slope, intercept, epoch = gradient_descent(x, y)
    plt.plot(x, slope * x + intercept, color='black', label='GD')
    print(f'Gradient Descent Function: \n y = {slope}x + {intercept}')
    print(f'Epochs = {epoch}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

    # plot our cost vs weights demonstrating GD algorithm
    plt.figure(figsize=(8, 6))
    plt.title('Cost vs Weight')
    plt.xlabel('Weight')
    plt.ylabel('Cost')
    plt.plot(weights, costs)
    plt.scatter(weights, costs)
    plt.show()

    # plot our cost vs biases demonstrating GD algorithm
    plt.figure(figsize=(8, 6))
    plt.title('Cost vs Biases')
    plt.xlabel('Bias')
    plt.ylabel('Cost')
    plt.plot(biases, costs)
    plt.scatter(biases, costs)
    plt.show()

    # Testing how correlated our data is:
    # create a pie chart of the partition
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
    plt.suptitle(f'Partition of {title} (R^2)')
    SSe, SSr = partition_ols(x, y)
    ax1.pie([SSr, SSe], labels=('Pred SS', 'Error SS'), autopct='%1.1f%%', colors=['#388E3C', '#B53737'], startangle=15)
    SSe, SSr = partition_gd(x, y)
    ax2.pie([SSr, SSe], labels=('Pred SS', 'Error SS'), autopct='%1.1f%%', colors=['#388E3C', '#B53737'], startangle=15)
    ax1.set_title('OLS')
    ax2.set_title('GD')
    plt.show()
