import matplotlib.pyplot as plt
import numpy as np

def plot_gaussian_dataset(data1, data2, m, b, title, color="blue"):
    """
    Plots a 2D array of Gaussian samples.
    Expects data in an N x 2 matrix format.
    """
    x1 = data1[:, 0]
    y1 = data1[:, 1]

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x1, y1, color="red", s=1, alpha=0.05)
    ax.scatter(x2, y2, color=color, s=1, alpha=0.05)
    ax.set_aspect('equal')
    
    # # ax.axline(boundary_x, boundary_y, color="red")
    x_line = np.linspace(-2.5, 8, 100)
    y_line = m*x_line + b
    ax.plot(x_line, y_line, color="red")

    ax.set_title(title)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(title)

def plot_gaussian_dataset_quadric(data1, data2, title, color="blue"):
    """
    Plots a 2D array of Gaussian samples.
    Expects data in an N x 2 matrix format.
    """
    x1 = data1[:, 0]
    y1 = data1[:, 1]

    x2 = data2[:, 0]
    y2 = data2[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x1, y1, color="red", s=1, alpha=0.05)
    ax.scatter(x2, y2, color=color, s=1, alpha=0.05)
    
    # plot decision boundary
    x_boundary = np.linspace(-5, 10, 400)
    y_boundary = np.linspace(-5, 10, 400)
    X, Y = np.meshgrid(x_boundary, y_boundary)

    LHS = (-0.5 * (X ** 2)) + (-0.5 * (Y ** 2)) + X + Y - 2.204
    RHS = (-(1/8) * (X ** 2)) + (-(1/16) * (Y ** 2)) + X + 0.5 * Y - 5.0895

    equal_r_l = LHS - RHS

    ax.contour(X, Y, equal_r_l, levels=[0], colors="red")
    
    ax.set_aspect('equal')
    ax.set_xlim(-5, 13.5)
    ax.set_ylim(-5, 13.5)
    ax.set_title(title)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(title)