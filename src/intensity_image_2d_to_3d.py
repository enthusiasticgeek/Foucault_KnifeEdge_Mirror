#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.ndimage import minimum_filter
import argparse

try:
    parser = argparse.ArgumentParser(description='Plot 3D surface and local maxima/minima of an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()

    # Load the image
    img = Image.open(args.image_path)
    img = img.convert('L')  # Convert to grayscale if the image is not in grayscale already

    # Get pixel intensities as a NumPy array
    pixel_intensity = np.array(img)

    # Normalize pixel intensities between 0 and 1
    normalized_intensity = (pixel_intensity - pixel_intensity.min()) / (pixel_intensity.max() - pixel_intensity.min())

    # Create meshgrid for x and y coordinates based on image dimensions
    x_dim, y_dim = img.size
    x = np.linspace(0, x_dim - 1, x_dim)
    y = np.linspace(0, y_dim - 1, y_dim)
    x, y = np.meshgrid(x, y)

    # Find local maxima positions using a maximum filter
    local_max = maximum_filter(normalized_intensity, size=100) == normalized_intensity

    # Find local minima positions using a minimum filter
    local_min = minimum_filter(normalized_intensity, size=100) == normalized_intensity

    # Get x, y coordinates of local maxima
    maxima_x = x[local_max]
    maxima_y = y[local_max]
    maxima_intensity = normalized_intensity[local_max]

    # Get x, y coordinates of local minima
    minima_x = x[local_min]
    minima_y = y[local_min]
    minima_intensity = normalized_intensity[local_min]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(x, y, normalized_intensity, cmap='viridis')

    # Plot local maxima as points on the surface
    ax.scatter(maxima_x, maxima_y, maxima_intensity, color='red', s=50, label='Local Maxima')

    # Plot local minima as points on the surface
    ax.scatter(minima_x, minima_y, minima_intensity, color='green', s=50, label='Local Minima')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')

    # Show the plot
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

