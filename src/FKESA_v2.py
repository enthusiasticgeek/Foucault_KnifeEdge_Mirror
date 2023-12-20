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
    cv2.line(image, (x, y - line_length), (x, y + line_length), color, thickness=2)

def draw_symmetrical_arc(image, x, y, r, start_angle, end_angle, color):
    # Define the center, axes lengths, start and end angles of the arc
    center = (x, y)
    axes = (r, r)
    angle = 0  # Angle is set to 0 for a symmetrical arc
    # Draw the arc on the image
    cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness=2)

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=1):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    text_y = max(text_y, text_size[1])  # Ensure the text doesn't go out of the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image



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
parser.add_argument('-rfc', '--retryFindMirror', type=int, default=1, help='Adjust Hough Transform search window (adaptive) and attempt to find Mirror. default 1')
parser.add_argument('-mdia', '--mirrorDiameterInches', type=float, default=6, help='Mirror diameter in inches. Default value is 6.0')
parser.add_argument('-mfl', '--mirrorFocalLengthInches', type=float, default=48, help='Mirror Focal Length in inches. Default value is 48.0')


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

    """
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
    """
    #The code attempts the circle detection process four times by shifting search window by e.g., 5, 10,-5,-10, etc.

    circles=None
    #Adaptive params mirror detection method
    if args.retryFindMirror == 1:
            # Define initial parameter values
            param1_initial = args.param1
            param2_initial = args.param2

            circle_found = False

            # Loop to retry with different parameters
            adjustments = [5, 10, 20, 30, -5, -10, -20, -30]
            for adjustment in adjustments:
                args.param1 = param1_initial + adjustment
                args.param2 = param2_initial + adjustment

                # Ensure param1 and param2 don't fall below certain thresholds
                args.param1 = max(10, args.param1)
                args.param2 = max(20, args.param2)

                print(f"Trying adaptive mirror detection with Hough Transform param1 {args.param1} and param2 {args.param2} and {adjustment}")

                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT, dp=1, minDist=args.minDist, param1=args.param1, param2=args.param2,
                    minRadius=args.minRadius, maxRadius=args.maxRadius
                )

                if circles is not None:
                    circle_found = True
                    print("Mirror found in retry attempt!")
                    break  # Exit the loop if circles are found with any parameter set

            if not circle_found:
                raise Exception("Hough Circle Transform didn't find any circles after trying multiple parameter combinations")


    #Simple params mirror detection method
    else:
            # Apply Hough Circle Transform with user-defined parameters
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT, dp=1, minDist=args.minDist, param1=args.param1, param2=args.param2,
                minRadius=args.minRadius, maxRadius=args.maxRadius
            )

            if circles is None:
                raise Exception("Hough Circle Transform didn't find any circles")


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
        #cv2.imshow('Cropped Image', cropped_gray_image)
        #cv2.waitKey(1000)


        # Get image dimensions
        height, width = cropped_image.shape[:2]

        # Define the center of the image
        center_x = width // 2
        center_y = height // 2

        # Define the number of zones
        num_zones = args.Zones

        # Create a blank mask
        mask = np.zeros((height, width), dtype=np.uint8)

        #pixels per zone
        pixels_per_zone = radius // num_zones
        print(pixels_per_zone)

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


        line_mark1 = 0 + (int(sorted_deltas[0][0]) * radius // num_zones)
        #line_mark2 = 0 + ((int(sorted_deltas[0][0])+1) * radius // num_zones)
        line_mark = int(int(line_mark1) + (int(pixels_per_zone)/2))
        #draw_symmetrical_line(cropped_image, center_x+line_mark, center_y, 20, color=(255,255,255))
        #draw_symmetrical_line(cropped_image, center_x-line_mark, center_y, 20, color=(255,255,255))


        # Define parameters for the arc
        start_angle = -20
        end_angle = 20
        color = (255, 255, 255)  # Blue color in BGR

        # Use the function to draw the symmetrical arc on the image
        draw_symmetrical_arc(cropped_image, center_x, center_y, line_mark1, start_angle, end_angle, color)
        draw_symmetrical_arc(cropped_image, center_x, center_y, line_mark1, start_angle+180, end_angle+180, color)
        

        draw_text(image, f"Radius: {radius}", (center_x-20,center_y+radius-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)

        cv2.imshow('Cropped Image', cropped_image)
        cv2.waitKey(1000)

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
