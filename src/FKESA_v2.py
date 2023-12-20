#!/usr/bin/env python3
#Author: Pratik M Tambe
#Date: Dec 18, 2023
#This method was proposed by Dr. Alin Tolea to me, Guy Brandenburg and Alan Tarica
#I have attempted to implement this
import matplotlib
# On some Linux systems may need to uncomment this.
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import numpy as np
import argparse
import pprint

def resize_image(image, max_width=640):
    height, width, _ = image.shape
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(image, (max_width, new_height))
    return image

def draw_symmetrical_line(image, x, y, line_length, color):
    cv2.line(image, (x, y - line_length), (x, y + line_length), color, thickness=1)


# Argument parser setup
parser = argparse.ArgumentParser(description='Detect largest circle in an image')
parser.add_argument('filename', help='Path to the image file')
parser.add_argument('-d', '--minDist', type=int, default=50, help='Minimum distance between detected circles. Default 50')
parser.add_argument('-p1', '--param1', type=int, default=20, help='First method-specific parameter. Default 30')
parser.add_argument('-p2', '--param2', type=int, default=60, help='Second method-specific parameter. Default 60')
parser.add_argument('-minR', '--minRadius', type=int, default=10, help='Minimum circle radius. Default 10')
parser.add_argument('-maxR', '--maxRadius', type=int, default=0, help='Maximum circle radius. Default 0')
parser.add_argument('-dwpz', '--displayWindowPeriodZones', type=int, default=100, help='Maximum period to wait in milliseconds between displaying zones. Default 100 milliseconds')
parser.add_argument('-bt', '--brightnessTolerance', type=int, default=10, help='Brightness Tolerance. Default 10')
parser.add_argument('-rad', '--roiAngleDegrees', type=int, default=10, help='ROI angle degrees. Default 10')
parser.add_argument('-z', '--Zones', type=int, default=50, help='Number of zones. Default 50')
parser.add_argument('-szfc', '--skipZonesFromCenter', type=int, default=10, help='Skip Number of zones from the center of the mirror. Default 10')

# Parse the arguments
args = parser.parse_args()

try:
    # Load the image using the provided filename
    image = cv2.imread(args.filename)
    filename_base =  os.path.basename(args.filename)
    
    if image is None:
        raise FileNotFoundError(f"File not found: {args.filename}")
 
    image = resize_image(image)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply Hough Circle Transform with provided parameters
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        args.minDist,
        param1=args.param1,
        param2=args.param2,
        minRadius=args.minRadius,
        maxRadius=args.maxRadius
    )

    # If circles are found
    if circles is not None:
        # Round the circle parameters and convert them to integers
        circles = np.uint16(np.around(circles))

        # Get the largest circle
        largest_circle = circles[0, :][0]  # Assuming the first circle is the largest

        # Get the center coordinates and radius of the largest circle
        center_x, center_y, radius = largest_circle

        # Mark the center of the largest circle on the image
        cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)

        # Calculate the bounding box coordinates
        top_left_x = abs(int(center_x) - int(radius))
        top_left_y = abs(int(center_y) - int(radius))
        bottom_right_x = int(center_x) + int(radius)
        bottom_right_y = int(center_y) + int(radius)

        # Ensure the coordinates are within the image boundaries
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        bottom_right_x = min(image.shape[1], bottom_right_x)
        bottom_right_y = min(image.shape[0], bottom_right_y)

        # Crop the image using the bounding box
        cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        cropped_gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        cropped_zones_image = cropped_gray_image.copy()

        # Apply Gaussian blur to reduce noise
        cropped_gray_image = cv2.GaussianBlur(cropped_gray_image, (5, 5), 0)
        cv2.imshow('Cropped Image', cropped_gray_image)
        cv2.waitKey(1000)


        # Get image dimensions
        height, width = cropped_image.shape[:2]

        # Define the center of the image
        center_x = width // 2
        center_y = height // 2

        # Define the number of zones
        num_zones = args.Zones

        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # List to store average intensities in each zone
        average_intensities_rhs = []
        average_intensities_lhs = []
        print("----RHS of mirror center zones and intensities----")
        # Iterate through each zone - R.H.S of the center of the mirror
        for zone in range(num_zones):
            # Reset the mask for each zone
            mask = np.zeros_like(cropped_gray_image, dtype=np.uint8)

            # Define radii for the current zone
            inner_radius = 0 + (zone * radius // num_zones)
            outer_radius = 0 + ((zone + 1) * radius // num_zones)

            # Define angles for the curves (+45 and -45 degrees)
            angle_45 = args.roiAngleDegrees
            shift_angle = 0

            # Create circles at specified radii and angles
            start_angle_positive = shift_angle - angle_45
            end_angle_positive = shift_angle + angle_45
            start_angle_negative = shift_angle - angle_45
            end_angle_negative = shift_angle + angle_45

            cv2.ellipse(mask, (center_x, center_y), (outer_radius, outer_radius), 0, start_angle_positive, end_angle_negative, 255, -1)
            cv2.ellipse(mask, (center_x, center_y), (inner_radius, inner_radius), 0, start_angle_positive, end_angle_negative, 0, -1)

            # Apply the mask to the grayscale image to get the custom ROI
            roi = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask)
            # Calculate the average intensity in the ROI
            average_intensity_rhs = (cv2.mean(roi)[0]*255)

            # Apply the mask to the grayscale image to get the custom ROI
            #roi = cropped_gray_image[mask > 0]  # Extract pixels within the mask
            # Calculate the average intensity in the ROI
            #average_intensity_rhs = np.mean(roi)

            average_intensities_rhs.append(average_intensity_rhs)
            print(f"Zone {zone + 1}: Average Intensity RHS = {average_intensity_rhs}")

            # Display the result (optional)
            #cv2.imshow('Image with Custom ROI', roi)
            #cv2.waitKey(args.displayWindowPeriodZones)


        print("----LHS of mirror center zones and intensities----")
        # Iterate through each zone - L.H.S of the center of the mirror
        for zone in range(num_zones):

            # Reset the mask for each zone
            mask = np.zeros_like(cropped_gray_image, dtype=np.uint8)

            # Define radii for the current zone
            inner_radius = 0 + (zone * radius // num_zones)
            outer_radius = 0 + ((zone + 1) * radius // num_zones)

            # Define angles for the curves (+45 and -45 degrees)
            angle_45 = args.roiAngleDegrees
            shift_angle = 180

            # Create circles at specified radii and angles
            start_angle_positive = shift_angle - angle_45
            end_angle_positive = shift_angle + angle_45
            start_angle_negative = shift_angle - angle_45
            end_angle_negative = shift_angle + angle_45

            cv2.ellipse(mask, (center_x, center_y), (outer_radius, outer_radius), 0, start_angle_positive, end_angle_negative, 255, -1)
            cv2.ellipse(mask, (center_x, center_y), (inner_radius, inner_radius), 0, start_angle_positive, end_angle_negative, 0, -1)

            # Apply the mask to the grayscale image to get the custom ROI
            roi = cv2.bitwise_and(cropped_gray_image, cropped_gray_image, mask=mask)
            # Calculate the average intensity in the ROI
            average_intensity_lhs = (cv2.mean(roi)[0]*255)

            # Apply the mask to the grayscale image to get the custom ROI
            #roi = cropped_gray_image[mask > 0]  # Extract pixels within the mask
            # Calculate the average intensity in the ROI
            #average_intensity_lhs = np.mean(roi)

            average_intensities_lhs.append(average_intensity_lhs)
            print(f"Zone {zone + 1}: Average Intensity LHS = {average_intensity_lhs}")

            # Display the result (optional)
            #cv2.imshow('Image with Custom ROI', roi)
            #cv2.waitKey(args.displayWindowPeriodZones)

        #"""
        deltas=[]
        #Check if the intensities match
        for zone in range(num_zones):
            #print(f"{zone}")
            if abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]) <= args.brightnessTolerance and zone > args.skipZonesFromCenter:
               print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
               difference = abs(average_intensities_rhs[zone] - average_intensities_lhs[zone])
               deltas.append((zone,difference))
        #"""
        """
        deltas=[]
        #Check if the intensities match
        for zone in range(num_zones):
               print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
               difference = abs(average_intensities_rhs[zone] - average_intensities_lhs[zone])
               deltas.append((zone,difference))

        """

        # Sort the deltas list by the second value (absolute differences), skipping when the second value is 0
        sorted_deltas = sorted(
            [t for t in deltas if t[0] != 0],  # Exclude tuples where the first value is 0
            key=lambda x: x[1]
        )

        sorted_deltas = sorted(deltas, key=lambda x: x[1])
        pprint.pprint(f"Zone match: {sorted_deltas[0][0]}")


        line_mark = 0 + (int(sorted_deltas[0][0]) * radius // num_zones)
        draw_symmetrical_line(cropped_image, center_x+line_mark, center_y, 20, color=(255,255,255))
        draw_symmetrical_line(cropped_image, center_x-line_mark, center_y, 20, color=(255,255,255))

        # Extract zones and deltas for plotting
        zones = [zone[0] for zone in deltas]
        differences = [zone[1] for zone in deltas]

        # Plotting the zones against their delta values
        plt.figure(figsize=(8, 6))
        plt.plot(zones, differences, marker='o', linestyle='-', color='blue')
        plt.title(f'Zones vs Delta Values for {filename_base} for {num_zones} zones within angle {args.roiAngleDegrees} Degrees')
        plt.xlabel('Zones')
        plt.ylabel('Delta Values')
        plt.grid(True)
        plt.xticks(zones)  # Set x-axis ticks to match zones
        plt.show()

        cv2.imwrite(args.filename + '.cropped.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.destroyAllWindows()
    else:
        raise ValueError("No circles detected.")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

