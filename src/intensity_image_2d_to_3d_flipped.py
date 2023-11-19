#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
import argparse

try:
    parser = argparse.ArgumentParser(description='Plot 3D surface and local maxima of an image')
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

    # Flip the intensity array for the flipped image along the y-axis
    flipped_intensity = normalized_intensity[::-1, :]

    # Find local maxima positions using a maximum filter
    local_max = maximum_filter(normalized_intensity, size=100) == normalized_intensity

    # Get x, y coordinates of local maxima
    maxima_x = x[local_max]
    maxima_y = y[local_max]
    maxima_intensity = normalized_intensity[local_max]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface for the original image
    ax.plot_surface(x, y, normalized_intensity, cmap='viridis', alpha=0.7)

    # Plot the flipped image
    ax.plot_surface(x, y, flipped_intensity, cmap='plasma', alpha=0.7)  # Flipped and superimposed image

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')

    # Show the plot
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")

