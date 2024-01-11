#!/usr/bin/env python3
#Author: Pratik M Tambe<enthusiasticgeek@gmail.com>
import matplotlib
# On some Linux systems may need to uncomment this.
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import numpy as np
import argparse

try:

        # Create argument parser
        parser = argparse.ArgumentParser(description='Plot parabolic mirror surface comparison')
        parser.add_argument('--radius', type=float, default=8, help='Radius of the mirror')
        parser.add_argument('--focal_length', type=float, default=75.8, help='Focal length of the mirror')
        parser.add_argument('--classic', action='store_true', help='Use classic parabolic equation')

        # Parse the arguments
        args = parser.parse_args()


        focal_length = args.focal_length
        radius = args.radius
        x = np.linspace(-focal_length, focal_length, 1000)

        # Equations
        y_simple = (1 / (4 * focal_length)) * (x ** 2)
        y_corrected = (1 / (4 * focal_length)) * (x ** 2) + (1 / (16 * (focal_length ** 3))) * (x ** 4)

        plt.figure(figsize=(8, 6))

        # Plotting in blue for the simple parabolic equation
        blue_curve = plt.plot(x, y_simple, linestyle='-', color='blue', label='Parabolic y=1/(4*f)*(r^2)   (Classic)')

        # Plotting in red for the corrected equation
        red_curve = plt.plot(x, y_corrected, linestyle='--', color='red', label='Parabolic y=1/(4*f)*(r^2)+1/(16*(f^3))*(r^4)   (T.H.Hussey)')

        if args.classic:

                # Marking the radius on the red curve with green dots (classic)
                plt.scatter(radius, 1 / (4 * focal_length) * (radius ** 2), color='green', s=100, label=f'+Radius ({radius} inches)', zorder=5)
                plt.scatter(-radius, 1 / (4 * focal_length) * (radius ** 2), color='green', s=100, label=f'- Radius ({radius} inches)', zorder=5)
        else:

                # Marking the radius on the red curve with green dots (T.H.Hussey)
                plt.scatter(radius, 1 / (4 * focal_length) * (radius ** 2) + (1 / (16 * (focal_length ** 3))) * (radius ** 4), color='green', s=100, label=f'+Radius ({radius} inches)', zorder=5)
                plt.scatter(-radius, 1 / (4 * focal_length) * (radius ** 2) + (1 / (16 * (focal_length ** 3))) * (radius ** 4), color='green', s=100, label=f'- Radius ({radius} inches)', zorder=5)

        # Marking the focal length on the y-axis with a yellow dot
        plt.scatter(0, focal_length, color='orange', s=100, label=f'Focal Length ({focal_length} inches)', zorder=5)

        if args.classic:
                # light rays (Classic)
                # Drawing lines from the orange dot to the green dots
                plt.plot([0, radius], [focal_length, 1 / (4 * focal_length) * (radius ** 2)], color='yellow', linestyle='-', linewidth=1)
                plt.plot([0, -radius], [focal_length, 1 / (4 * focal_length) * (radius ** 2)], color='yellow', linestyle='-', linewidth=1)
                # Drawing a line starting from the green dot and extending upwards
                plt.plot([radius, radius], [1 / (4 * focal_length) * (radius ** 2), 100], color='yellow', linestyle='-', linewidth=1, label='x = Radius')
                # Drawing a line starting from the green dot on the negative axis and extending upwards
                plt.plot([-radius, -radius], [1 / (4 * focal_length) * (radius ** 2), 100], color='yellow', linestyle='-', linewidth=1, label='x = -Radius')

        else:
                # light rays (T.H.Hussey)
                # Drawing lines from the orange dot to the green dots
                plt.plot([0, radius], [focal_length, 1 / (4 * focal_length) * (radius ** 2)+ (1 / (16 * (focal_length ** 3))) * (radius ** 4)], color='yellow', linestyle='-', linewidth=1)
                plt.plot([0, -radius], [focal_length, 1 / (4 * focal_length) * (radius ** 2)+ (1 / (16 * (focal_length ** 3))) * (radius ** 4)], color='yellow', linestyle='-', linewidth=1)
                # Drawing a line starting from the green dot and extending upwards
                plt.plot([radius, radius], [1 / (4 * focal_length) * (radius ** 2)+ (1 / (16 * (focal_length ** 3))) * (radius ** 4), 100], color='yellow', linestyle='-', linewidth=1, label='x = Radius')
                # Drawing a line starting from the green dot on the negative axis and extending upwards
                plt.plot([-radius, -radius], [1 / (4 * focal_length) * (radius ** 2)+ (1 / (16 * (focal_length ** 3))) * (radius ** 4), 100], color='yellow', linestyle='-', linewidth=1, label='x = -Radius')

        plt.title('Parabolic Mirror Surface Comparison')
        plt.xlabel('radius of mirror')
        plt.ylabel('distance')
        plt.legend()
        plt.grid(True)
        plt.show()

except Exception as e:
    print("An error occurred:", e)
