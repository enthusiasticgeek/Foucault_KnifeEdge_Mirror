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


class FKESABuilder:
    def __init__(self):
        self.args = {
            'gammaCorrection': 0,
            'minDist': 50,
            'param1': 25,
            'param2': 60,
            'minRadius': 10,
            'maxRadius': 0,
            'brightnessTolerance': 10,
            'roiAngleDegrees': 10,
            'zones': 50,
            'mirrorDiameterInches': 6,
            'mirrorFocalLengthInches': 48,
            'gradientIntensityChange': 3,
            'skipZonesFromCenter': 10
            # Include default values for other parameters here
        }

    def with_folder(self, folder_path=''):
        self.args['folder'] = folder_path or self.args['folder']
        return self

    def with_param(self, param_name, value):
        self.args[param_name] = value
        return self

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        else:
            print(f"Folder '{folder_path}' already exists.")
        return self

    def resize_image(self, image, max_width=640):
        height, width, _ = image.shape
        if width > max_width:
            ratio = max_width / width
            new_height = int(height * ratio)
            return cv2.resize(image, (max_width, new_height))
        return image

    # Ref: https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    def adjust_gamma(self, image, gamma=0):
            # build a lookup table mapping the pixel values [0, 255] to
            # their adjusted gamma values
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")
            # apply gamma correction using the lookup table
            return cv2.LUT(image, table)

    def draw_symmetrical_line(self, image, x, y, line_length, color):
        cv2.line(image, (x, y - line_length), (x, y + line_length), color, thickness=2)

    def draw_symmetrical_arc(self, image, x, y, r, start_angle, end_angle, color):
        # Define the center, axes lengths, start and end angles of the arc
        center = (x, y)
        axes = (r, r)
        angle = 0  # Angle is set to 0 for a symmetrical arc
        # Draw the arc on the image
        cv2.ellipse(image, center, axes, angle, start_angle, end_angle, color, thickness=2)

    def draw_text(self, image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=1):
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x, text_y = position
        text_y = max(text_y, text_size[1])  # Ensure the text doesn't go out of the image
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        return image


    def build(self, image):
        try:
            cropped_image=None
            # Your existing logic, but using class methods with self.
            if image is not None:
                if self.args['gammaCorrection'] > 0.0:
                    image = self.adjust_gamma(image, gamma=self.args['gammaCorrection'])

                image = self.resize_image(image)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

                circles = None
                # Loop for adaptive mirror detection with different parameters
                adjustments = [5, 10, 20, 30, -5, -10, -20, -30]
                for adjustment in adjustments:
                    self.args['param1'] = self.args['param1'] + adjustment
                    self.args['param2'] = self.args['param2'] + adjustment

                    # Ensure param1 and param2 don't fall below certain thresholds
                    self.args['param1'] = max(10, self.args['param1'])
                    self.args['param2'] = max(20, self.args['param2'])

                    print(f"Trying adaptive mirror detection with Hough Transform "
                          f"param1 {self.args['param1']} and param2 {self.args['param2']} and {adjustment}")

                    circles = cv2.HoughCircles(
                        blurred,
                        cv2.HOUGH_GRADIENT, dp=1, minDist=self.args['minDist'], param1=self.args['param1'],
                        param2=self.args['param2'], minRadius=self.args['minRadius'], maxRadius=self.args['maxRadius']
                    )

                    if circles is not None:
                        print("Mirror found in retry attempt!")
                        break  # Exit the loop if circles are found with any parameter set

                if circles is None:
                    raise Exception("Hough Mirror Transform didn't find any circles after trying multiple parameter combinations")

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

                    # Flip the cropped image horizontally
                    flipped_cropped_gray_image = cv2.flip(cropped_gray_image, 1)

                    # image like phi
                    phi_image = cv2.absdiff(cropped_gray_image, flipped_cropped_gray_image)

                    # Apply a filter (e.g., GaussianBlur) to phi_image
                    filtered_image = cv2.GaussianBlur(phi_image, (5, 5), 0)  # You can choose different filter types and kernel sizes

                    # Increase the contrast of the filtered image
                    alpha = 2.0  # Contrast control (1.0-3.0, 1.0 is normal)
                    beta = 0    # Brightness control (0-100, 0 is normal)
                    phi_final_image = cv2.convertScaleAbs(filtered_image, alpha=alpha, beta=beta)

                    # Define a kernel for dilation
                    #kernel = np.ones((3, 3), np.uint8)  # Adjust the size and shape as needed

                    # Apply sharpening
                    kernel = np.array([[-1,-1,-1],
                                   [-1,9,-1],
                                   [-1,-1,-1]])  # Sharpening kernel

                    # Perform dilation on the image
                    phi_final_image = cv2.dilate(phi_final_image, kernel, iterations=1)  # Adjust the number of iterations as needed

                    # Get image dimensions
                    height, width = cropped_image.shape[:2]

                    # Define the center of the image
                    center_x = width // 2
                    center_y = height // 2

                    # Define the number of zones
                    num_zones = self.args['zones']

                    if num_zones > 50:
                        print("WARNING!!! - Number of zones exceed 50. Limiting to 50.")
                        num_zones = 50
                    elif num_zones < 30:
                        print("WARNING!!! - Number of zones are less than 30. Limiting to 30.")
                        num_zones = 30

                    # Create a blank mask
                    mask = np.zeros((height, width), dtype=np.uint8)

                    #pixels per zone
                    pixels_per_zone = float(radius / num_zones)
                    print("pixels per zone are ", pixels_per_zone)

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
                        angle_45 = self.args['roiAngleDegrees']
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
                        angle_45 = self.args['roiAngleDegrees']
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

                   # Initialize an empty list to store the zonal_intensity_deltas
                    zonal_intensity_deltas_rhs = []

                    # Iterate through the list to find zonal_intensity_deltas between consecutive elements
                    for i in range(len(average_intensities_rhs) - 1):
                        diff = round(average_intensities_rhs[i + 1] - average_intensities_rhs[i], 2)
                        zonal_intensity_deltas_rhs.append(diff)

                    print(f"Zonal intensity difference between consecutive elements rhs: {zonal_intensity_deltas_rhs} and {len(zonal_intensity_deltas_rhs)}")

                   # Initialize an empty list to store the zonal_intensity_deltas
                    zonal_intensity_deltas_lhs = []

                    # Iterate through the list to find zonal_intensity_deltas between consecutive elements
                    for i in range(len(average_intensities_lhs) - 1):
                        diff = round(average_intensities_lhs[i + 1] - average_intensities_lhs[i], 2)
                        zonal_intensity_deltas_lhs.append(diff)

                    print(f"Zonal intensity difference between consecutive elements lhs: {zonal_intensity_deltas_lhs} and {len(zonal_intensity_deltas_lhs)}")

                    null_zone_rhs_possibility = False
                    for diff in zonal_intensity_deltas_rhs:
                        if abs(diff) > self.args['gradientIntensityChange']:
                            null_zone_rhs_possibility = True
                            break  # No need to continue checking if one element meets the condition
                    print("Null Zone RHS Possibility:", null_zone_rhs_possibility)

                    null_zone_lhs_possibility = False
                    for diff in zonal_intensity_deltas_lhs:
                        if abs(diff) > self.args['gradientIntensityChange']:
                            null_zone_lhs_possibility = True
                            break  # No need to continue checking if one element meets the condition
                    print("Null Zone LHS Possibility:", null_zone_lhs_possibility)

                    #"""
                    deltas=[]
                    #Check if the intensities match
                    for zone in range(num_zones):
                        #print(f"{zone}")
                        if abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]) <= self.args['brightnessTolerance'] and zone > self.args['skipZonesFromCenter']:
                           print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
                           difference = abs(average_intensities_rhs[zone] - average_intensities_lhs[zone])
                           deltas.append((zone,difference))
                    if deltas:
                        #"""
                        """
                        deltas=[]
                        #Check if the intensities match
                        for zone in range(num_zones):
                               print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
                               difference = abs(average_intensities_rhs[zone] - average_intensities_lhs[zone])
                               deltas.append((zone,difference))

                        """

                        # Sort the deltas list by the second value (absolute differences), skipping when the first value is 0
                        #sorted_deltas = sorted(
                        #    [t for t in deltas if t[0] != 0],  # Exclude tuples where the first value is 0
                        #    key=lambda x: x[1]
                        #)

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

                        if null_zone_lhs_possibility == True and null_zone_lhs_possibility == True:
                            color = (255, 255, 255)  # White color in BGR
                            self.draw_text(image, f"NULL Zones found", (center_x-20,center_y+radius-80), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 255, 0), thickness=1)
                        else:
                            color = (0, 0, 255)  # Red color in BGR
                            self.draw_text(image, f"NULL Zones not found", (center_x-20,center_y+radius-80), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 0, 255), thickness=1)

                        # Use the function to draw the symmetrical arc on the image
                        self.draw_symmetrical_arc(cropped_image, center_x, center_y, line_mark1, start_angle, end_angle, color)
                        self.draw_symmetrical_arc(cropped_image, center_x, center_y, line_mark1, start_angle+180, end_angle+180, color)
                        
                        zone_pixels = int(pixels_per_zone * int(sorted_deltas[0][0]))
                        zone_inches = float(float(zone_pixels/radius)*self.args['mirrorDiameterInches'])/2
                        self.draw_text(image, f"Radius: {radius} pixels ", (center_x-20,center_y+radius-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(image, f"Zone: {zone_pixels:.0f} pixels or {zone_inches:.4f} \"", (center_x-20,center_y+radius-40), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(image, f"Zone match: {sorted_deltas[0][0]} zone of total {num_zones} zones", (center_x-20,center_y+radius-60), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(image, f"Mirror Diameter: {self.args['mirrorDiameterInches']} \"", (center_x-20,center_y-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(image, f"Mirror Focal Length: {self.args['mirrorFocalLengthInches']} \"", (center_x-20,center_y-40), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)

                        #cv2.imshow('Cropped Image', cropped_image)
                        #cv2.waitKey(args.displayWindowPeriod)

                        # Extract zones and deltas for plotting
                        zones = [zone[0] for zone in deltas]
                        differences = [zone[1] for zone in deltas]

                    else:
                         print("No zones have matching intensities!")
                    # Return the cropped image
                    return cropped_image

            else:
                raise Exception("Image is not valid!")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

"""
# Example usage
builder = FKESABuilder()
builder.with_folder('output_folder')
builder.with_param('gammaCorrection',0)
builder.with_param('minDist', 50)
builder.with_param('param1', 25)
builder.with_param('param2', 60)
builder.with_param('minRadius', 10)
builder.with_param('maxRadius', 0)
builder.with_param('brightnessTolerance', 10)
builder.with_param('roiAngleDegrees', 10)
builder.with_param('zones', 50)
builder.with_param('mirrorDiameterInches', 6)
builder.with_param('mirrorFocalLengthInches', 48)
builder.with_param('gradientIntensityChange', 3)
builder.with_param('skipZonesFromCenter', 10)
# ... Include other parameters as needed

# Build and execute the operation
fkesa_detector_image = builder.build(image)
"""
