#!/usr/bin/env python3
#Author: Pratik M Tambe <enthusiasticgeek@gmail.com>
#Date: Dec 18, 2023
#This method was proposed by Dr. Alin Tolea to me, Guy Brandenburg and Alan Tarica
#I have attempted to implement this
import matplotlib
# On some Linux systems may need to uncomment this.
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import os
import cv2
import numpy as np
import argparse
import pprint
import csv
import platform
import sys
import time
import math

# Get the user's home directory
home_dir = os.path.expanduser("~")
#Example Specify the relative path from the home directory
#image_path = os.path.join(home_dir, 'Desktop', 'fkesa_v2.bmp')
# Perform the image write operation
#if not cv2.imwrite(image_path, img2):
#    raise Exception("Could not write image")



class FKESABuilder:
    def __init__(self):
        self.args = {
            'folder': 'fkesa_v2_default',
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
            'skipZonesFromCenter': 10,
            'csv_filename': 'fkesa_v2_default.csv',
            'append_to_csv': False,
            'step_size': 0.010,
            'user_text': '',
            'debug': False,
            'adaptive_find_mirror': False,
            'start_point': None,
            'end_point': None,
            'radius_of_points': None,
            'enable_disk_rwx_operations': False
            # Include default values for other parameters here
        }
        stale_image=False

    def debug_print(self, *args):
        if self.args['debug']:
            for arg in args:
                if isinstance(arg, str):
                    print(arg)
                else:
                    # If arg is not a string, print it as an exception
                    print("An exception occurred:", arg)

    def with_folder(self, folder_path=''):
        self.args['folder'] = folder_path or self.args['folder']
        self.create_folder_if_not_exists(folder_path)
        return self

    def with_param(self, param_name, value):
        self.args[param_name] = value
        return self

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            self.debug_print(f"Folder '{folder_path}' created successfully.")
        else:
            self.debug_print(f"Folder '{folder_path}' already exists.")
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

    def fill_image_boundary(self, image):
        desired_width, desired_height = 640, 480
        height, width = image.shape[:2]

        # Create a canvas of the desired size
        canvas = np.zeros((desired_height, desired_width, 3), dtype=np.uint8)

        # Calculate where to place the image on the canvas
        start_x = (desired_width - width) // 2
        start_y = (desired_height - height) // 2

        # Paste the image onto the canvas
        canvas[start_y:start_y+height, start_x:start_x+width] = image

        return canvas


    def current_timestamp(self):
        now = datetime.now()
        #milliseconds = round(now.timestamp() * 1000)
        #timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S") + f".{milliseconds:03d}"
        microseconds = round(now.microsecond / 1000)  # Convert microseconds to milliseconds
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S") + f".{microseconds:03d}"
        return timestamp_str

    # data is of format - writer.writerow(['X [pixels], Y [pixels], INTENSITY [0-255]'])
    def write_csv(self, data):
            csv_filename = self.args['csv_filename']
            csv_file = os.path.join(self.args['folder'], csv_filename)
            is_empty = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0

            # If the file is empty or doesn't exist, write the header
            if is_empty:
                try:
                    with open(csv_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            '------ Timestamp ------',
                            'Mirror X PX',
                            'Mirror Y PX',
                            'Mirror Rad PX',
                            'Zones',
                            'Match',
                            'Match PX',
                            'Match IN',
                            'Mirror DIA IN',
                            'Mirror FL IN',
                            'Step Size'
                        ])  # Replace with your column names
                except Exception as e:
                    print(f"Error writing headers: {e}")

            # Mode is append
            try:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)
            except Exception as e:
                print(f"Error writing data: {e}")

    def draw_circle(self, image, start_point, end_point, radius_of_points):
        center = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
        #radius = min(abs(start_point[0] - end_point[0]), abs(start_point[1] - end_point[1])) // 2
        radius = radius_of_points
        cv2.circle(image, center, radius, (0, 0, 255), thickness=2)

    def get_circle_from_bounding_box(self, start_point, end_point , x_offset, y_offset, radius_of_points):
        # Extract coordinates of start and end points
        x1, y1 = start_point
        x2, y2 = end_point
        x1 = abs(x_offset-x1)
        x2 = abs(x_offset-x2)
        y1 = abs(y_offset-y1)
        y2 = abs(y_offset-y2)
        # Calculate center of bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Calculate radius of the circle
        #radius = math.sqrt(int(x2 - x1)**2 + int(y2 - y1)**2) / 2
        radius = radius_of_points
        return int(center_x), int(center_y), int(radius)    

    def crop_circle_from_image(self, image):
        start_point = self.args['start_point']
        end_point = self.args['end_point']
        radius_of_points = self.args['radius_of_points']

        if start_point and end_point:
            center_x, center_y, radius = self.get_circle_from_bounding_box(start_point, end_point,0,0,radius_of_points)
            # Calculate bounding box coordinates for the circular region
            x1 = int(center_x - radius)
            y1 = int(center_y - radius)
            x2 = int(center_x + radius)
            y2 = int(center_y + radius)
            # Crop circular region from the image
            cropped_circle = image[y1:y2, x1:x2].copy()
            return cropped_circle
        else:
            return None

    def build_manual(self, image):
        try:
            cropped_image=None
            mask_ret=None
            # Your existing logic, but using class methods with self.
            if image is not None:
                if self.args['gammaCorrection'] > 0.0:
                    image = self.adjust_gamma(image, gamma=self.args['gammaCorrection'])

                image = self.resize_image(image)

                # Define the dimensions of the mask
                mask_width = image.shape[1]
                mask_height = image.shape[0]

                # Create a transparent mask with an alpha channel
                mask_ret = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)
                mask_ret[:, :, 3] = 0  # Full transparency initially

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
                #print(self.args['start_point'])
                #print(self.args['end_point'])
                #return None, None

                if self.args['start_point'] and self.args['end_point'] and self.args['radius_of_points']:
                    # Get the center coordinates and radius of the largest circle
                    center_x, center_y, radius = self.get_circle_from_bounding_box(self.args['start_point'],self.args['end_point'],0,480,self.args['radius_of_points'])
                
                    #print("=======================")
                    #print(center_x, center_y, radius)
                    #print("=======================")

                    #self.debug_print(f"LARGEST CIRCLE : {center_x},{center_y},{radius}")
                    # backup copies to be used in csv later
                    center_x_orig = center_x
                    center_y_orig = center_y
                    radius_orig = radius

                    # Mark the center of the largest circle on the image
                    cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)
                    #cv2.circle(image, (int(center_x), int(center_y)), int(radius), (0, 255, 0), 2)

                    """ 
                    top_left_x = abs(int(self.args['start_point'][0]))
                    top_left_y = abs(int(self.args['start_point'][1]))
                    bottom_right_x = int(self.args['end_point'][0])
                    bottom_right_y = int(self.args['end_point'][1])

                    """ 
                    # Calculate the bounding box coordinates
                    top_left_x = int(center_x - radius)
                    top_left_y = int(center_y - radius)
                    bottom_right_x = int(center_x + radius)
                    bottom_right_y = int(center_y + radius)

                    # Ensure the coordinates are within the image boundaries
                    top_left_x = max(0, top_left_x)
                    top_left_y = max(0, top_left_y)
                    bottom_right_x = min(image.shape[1], bottom_right_x)
                    bottom_right_y = min(image.shape[0], bottom_right_y)


                    """
                    # Calculate the bounding box coordinates
                    top_left_x = max(0, int(center_x - radius))
                    top_left_y = max(0, int(center_y - radius))
                    bottom_right_x = min(image.shape[1] - 1, int(center_x + radius))
                    bottom_right_y = min(image.shape[0] - 1, int(center_y + radius))
                    """

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
                        self.debug_print("WARNING!!! - Number of zones exceed 50. Limiting to 50.")
                        num_zones = 50
                    elif num_zones < 30:
                        self.debug_print("WARNING!!! - Number of zones are less than 30. Limiting to 30.")
                        num_zones = 30

                    # Create a blank mask
                    mask = np.zeros((height, width), dtype=np.uint8)

                    #pixels per zone
                    pixels_per_zone = float(radius / num_zones)
                    #print("pixels per zone are ", pixels_per_zone)

                    # List to store average intensities in each zone
                    average_intensities_rhs = []
                    average_intensities_lhs = []
                    self.debug_print("----RHS of mirror center zones and intensities----")
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
                        self.debug_print(f"Zone {zone + 1}: Average Intensity RHS = {average_intensity_rhs}")

                        # Display the result (optional)
                        #cv2.imshow('Image with Custom ROI', roi)
                        #cv2.waitKey(args.displayWindowPeriodZones)


                    self.debug_print("----LHS of mirror center zones and intensities----")
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
                        self.debug_print(f"Zone {zone + 1}: Average Intensity LHS = {average_intensity_lhs}")

                        # Display the result (optional)
                        #cv2.imshow('Image with Custom ROI', roi)
                        #cv2.waitKey(args.displayWindowPeriodZones)

                   # Initialize an empty list to store the zonal_intensity_deltas
                    zonal_intensity_deltas_rhs = []

                    # Iterate through the list to find zonal_intensity_deltas between consecutive elements
                    for i in range(len(average_intensities_rhs) - 1):
                        diff = round(average_intensities_rhs[i + 1] - average_intensities_rhs[i], 2)
                        zonal_intensity_deltas_rhs.append(diff)

                    self.debug_print(f"Zonal intensity difference between consecutive elements rhs: {zonal_intensity_deltas_rhs} and {len(zonal_intensity_deltas_rhs)}")

                   # Initialize an empty list to store the zonal_intensity_deltas
                    zonal_intensity_deltas_lhs = []

                    # Iterate through the list to find zonal_intensity_deltas between consecutive elements
                    for i in range(len(average_intensities_lhs) - 1):
                        diff = round(average_intensities_lhs[i + 1] - average_intensities_lhs[i], 2)
                        zonal_intensity_deltas_lhs.append(diff)

                    self.debug_print(f"Zonal intensity difference between consecutive elements lhs: {zonal_intensity_deltas_lhs} and {len(zonal_intensity_deltas_lhs)}")

                    null_zone_rhs_possibility = False
                    for diff in zonal_intensity_deltas_rhs:
                        if abs(diff) > self.args['gradientIntensityChange']:
                            null_zone_rhs_possibility = True
                            break  # No need to continue checking if one element meets the condition
                    self.debug_print("Null Zone RHS Possibility:", null_zone_rhs_possibility)

                    null_zone_lhs_possibility = False
                    for diff in zonal_intensity_deltas_lhs:
                        if abs(diff) > self.args['gradientIntensityChange']:
                            null_zone_lhs_possibility = True
                            break  # No need to continue checking if one element meets the condition
                    self.debug_print("Null Zone LHS Possibility:", null_zone_lhs_possibility)

                    #"""
                    deltas=[]
                    #Check if the intensities match
                    for zone in range(num_zones):
                        #print(f"{zone}")
                        if abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]) <= self.args['brightnessTolerance'] and zone > self.args['skipZonesFromCenter']:
                           self.debug_print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
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
                        #pprint.pprint(f"Zone match: {sorted_deltas[0][0]}")

                        line_mark1 = 0 + (int(sorted_deltas[0][0]) * radius // num_zones)
                        #line_mark2 = 0 + ((int(sorted_deltas[0][0])+1) * radius // num_zones)
                        line_mark = int(int(line_mark1) + (int(pixels_per_zone)/2))
                        #draw_symmetrical_line(cropped_image, center_x+line_mark, center_y, 20, color=(255,255,255))
                        #draw_symmetrical_line(cropped_image, center_x-line_mark, center_y, 20, color=(255,255,255))


                        # Define parameters for the arc
                        start_angle = -20
                        end_angle = 20

                        """
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
                        #cv2.waitKey(100)
                        """

                        # Extract zones and deltas for plotting
                        zones = [zone[0] for zone in deltas]
                        differences = [zone[1] for zone in deltas]

                        #============================ Begin:  do same thing on mask ==============================
                        if null_zone_lhs_possibility == True and null_zone_lhs_possibility == True:
                            color = (255, 255, 255)  # White color in BGR
                            self.draw_text(mask_ret, f"NULL Zones found", (center_x-20+top_left_x,center_y+radius-80+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 255, 0), thickness=1)
                        else:
                            color = (0, 0, 255)  # Red color in BGR
                            self.draw_text(mask_ret, f"NULL Zones not found", (center_x-20+top_left_x,center_y+radius-80+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 0, 255), thickness=1)

                        # Use the function to draw the symmetrical arc on the image
                        #self.draw_symmetrical_arc(mask_ret, center_x, center_y, line_mark1, start_angle, end_angle, color)
                        #self.draw_symmetrical_arc(mask_ret, center_x, center_y, line_mark1, start_angle+180, end_angle+180, color)
                        
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, line_mark1, start_angle, end_angle, color)
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, line_mark1, start_angle+180, end_angle+180, color)
 

                        zone_pixels = int(pixels_per_zone * int(sorted_deltas[0][0]))
                        zone_inches = float(float(zone_pixels/radius_orig)*self.args['mirrorDiameterInches'])/2
                        self.draw_text(mask_ret, f"Radius: {radius} pixels ", (center_x-20+top_left_x,center_y+radius-20+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Zone: {zone_pixels:.0f} pixels or {zone_inches:.4f} \"", (center_x-20+top_left_x,center_y+radius-40+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Zone match: {sorted_deltas[0][0]} zone of total {num_zones} zones", (center_x-20+top_left_x,center_y+radius-60+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Mirror Diameter: {self.args['mirrorDiameterInches']} \"", (center_x-20+top_left_x,center_y-20+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Mirror Focal Length: {self.args['mirrorFocalLengthInches']} \"", (center_x-20+top_left_x,center_y-40+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"{self.current_timestamp()}", (0,0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1)
                        self.stale_image = False
                        #============================ End: do same thing on mask ==============================
 
                        # Append to CSV if set
                        if self.args['append_to_csv']:
                            csv_data=[
                                self.current_timestamp(),
                                center_x_orig,
                                center_y_orig,
                                radius_orig,
                                num_zones,
                                int(sorted_deltas[0][0]),
                                zone_pixels,
                                round(zone_inches,3),
                                self.args['mirrorDiameterInches'],
                                self.args['mirrorFocalLengthInches'],
                                self.args['step_size']
                            ]
                            self.write_csv(csv_data)
                            #self.write_csv(','.join(map(str,csv_data)))
                            #csv_line = ','.join(str(item).replace('"', '') for item in csv_data)
                            #self.write_csv(csv_line)

                    else:
                         self.debug_print("No zones have matching intensities!")
                         self.stale_image = True
                    # Check if the image size is smaller than 640x480
                    
                    """
                    if cropped_image.shape[0] < 480 or cropped_image.shape[1] < 640:
                        # Fill the boundary
                        cropped_image = self.fill_image_boundary(cropped_image)
                    """
                    # Return the cropped image
                    # Overlay the mask on the image
                    result = np.copy(image)
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # Convert image to 4 channels (BGR + Alpha)

                    # Blend the mask with the image
                    alpha = 1.0  # Adjust the alpha blending factor (0.0 - fully transparent, 1.0 - fully opaque)
                    cv2.addWeighted(mask_ret, alpha, result, 1.0, 0, result)

                    # Calculate the position to place the text at the center of the image
                    result_center_x = result.shape[1] // 2
                    result_center_y = result.shape[0] // 2
                    # Draw the user text on the result image near center
                    self.draw_text(result, self.args['user_text'], color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, position=(result_center_x-20, result_center_y-20), thickness=1)

                    cv2.rectangle(result, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color=(0, 255, 0), thickness=2)

                    #Took measurement - Hence save the image
                    if self.args['append_to_csv'] and self.stale_image == False and self.args['enable_disk_rwx_operations']:
                       analyzed_jpg_filename = self.args['csv_filename']+self.current_timestamp()+'.jpg'
                       #analyzed_jpg_file = os.path.join(self.args['folder'], analyzed_jpg_filename)
                       #cv2.imwrite(analyzed_jpg_file, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
                       if platform.system() == "Linux":
                           analyzed_jpg_file = os.path.join(self.args['folder'], analyzed_jpg_filename)
                           if not cv2.imwrite(analyzed_jpg_file, result, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                              raise Exception("Could not write/save image")
                       elif platform.system() == "Windows":
                           save_directory = os.path.join(home_dir, 'FKESAv2Images')
                           os.makedirs(save_directory, exist_ok=True)
                           timestamp = int(time.time())  # Get the current timestamp
                           filename = f"FKESA_v2_{timestamp}.jpg"  # Generate a filename with the timestamp
                           image_path = os.path.join(save_directory,filename)
                           self.debug_print("********************************************************************")
                           self.debug_print(image_path)                           
                           self.debug_print("********************************************************************")
                           #sys.exit(1)
                           #image_path = os.path.join(home_dir, 'Desktop', analyzed_jpg_filename)
                           if not cv2.imwrite(image_path, result):
                              raise Exception("Could not write/save image")

                    #cv2.imwrite("some_file.jpg", cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

                    return cropped_image, result

            else:
                raise Exception("Image is not valid!")

        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


    def build_auto(self, image):
        try:
            cropped_image=None
            mask_ret=None
            # Your existing logic, but using class methods with self.
            if image is not None:
                if self.args['gammaCorrection'] > 0.0:
                    image = self.adjust_gamma(image, gamma=self.args['gammaCorrection'])

                image = self.resize_image(image)

                # Define the dimensions of the mask
                mask_width = image.shape[1]
                mask_height = image.shape[0]

                # Create a transparent mask with an alpha channel
                mask_ret = np.zeros((mask_height, mask_width, 4), dtype=np.uint8)
                mask_ret[:, :, 3] = 0  # Full transparency initially

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

                circles = None
                
                adjustments = [0]
                if self.args['adaptive_find_mirror']:
                   # Loop for adaptive mirror detection with different parameters
                   adjustments = [5, 10, 20, 30, -5, -10, -20, -30]
                for adjustment in adjustments:
                    self.args['param1'] = int(self.args['param1']) + adjustment
                    self.args['param2'] = int(self.args['param2']) + adjustment

                    # Ensure param1 and param2 don't fall below certain thresholds
                    self.args['param1'] = max(10, self.args['param1'])
                    self.args['param2'] = max(20, self.args['param2'])

                    self.debug_print(f"Trying adaptive mirror detection with Hough Transform "
                          f"param1 {self.args['param1']} and param2 {self.args['param2']} and {adjustment}")
                    self.debug_print(f"min distance ->>> {self.args['minDist']} , {self.args['param1']} , {self.args['param2']}, {self.args['minRadius']}, {self.args['maxRadius']}")

                    circles = cv2.HoughCircles(
                        blurred,
                        cv2.HOUGH_GRADIENT, dp=1, minDist=int(self.args['minDist']), param1=int(self.args['param1']),
                        param2=int(self.args['param2']), minRadius=int(self.args['minRadius']), maxRadius=int(self.args['maxRadius'])
                    )

                    if circles is not None:
                        self.debug_print("Mirror found in retry attempt!")
                        break  # Exit the loop if circles are found with any parameter set

                if circles is None:
                    self.stale_image = True
                    raise Exception("Hough Mirror Transform didn't find any circles after trying multiple parameter combinations")

                # If circles are found
                if circles is not None:
                    # Round the circle parameters and convert them to integers
                    circles = np.uint16(np.around(circles))

                    # Get the largest circle
                    largest_circle = circles[0, :][0]  # Assuming the first circle is the largest

                    # Get the center coordinates and radius of the largest circle
                    center_x, center_y, radius = largest_circle
                    self.debug_print(f"LARGEST CIRCLE : {center_x},{center_y},{radius}")
                    # backup copies to be used in csv later
                    center_x_orig = center_x
                    center_y_orig = center_y
                    radius_orig = radius

                    # Mark the center of the largest circle on the image
                    cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)

                    # Calculate the bounding box coordinates
                    top_left_x = int(center_x - radius)
                    top_left_y = int(center_y - radius)
                    bottom_right_x = int(center_x + radius)
                    bottom_right_y = int(center_y + radius)

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
                        self.debug_print("WARNING!!! - Number of zones exceed 50. Limiting to 50.")
                        num_zones = 50
                    elif num_zones < 30:
                        self.debug_print("WARNING!!! - Number of zones are less than 30. Limiting to 30.")
                        num_zones = 30

                    # Create a blank mask
                    mask = np.zeros((height, width), dtype=np.uint8)

                    #pixels per zone
                    pixels_per_zone = float(radius / num_zones)
                    #print("pixels per zone are ", pixels_per_zone)

                    # List to store average intensities in each zone
                    average_intensities_rhs = []
                    average_intensities_lhs = []
                    self.debug_print("----RHS of mirror center zones and intensities----")
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
                        self.debug_print(f"Zone {zone + 1}: Average Intensity RHS = {average_intensity_rhs}")

                        # Display the result (optional)
                        #cv2.imshow('Image with Custom ROI', roi)
                        #cv2.waitKey(args.displayWindowPeriodZones)


                    self.debug_print("----LHS of mirror center zones and intensities----")
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
                        self.debug_print(f"Zone {zone + 1}: Average Intensity LHS = {average_intensity_lhs}")

                        # Display the result (optional)
                        #cv2.imshow('Image with Custom ROI', roi)
                        #cv2.waitKey(args.displayWindowPeriodZones)

                   # Initialize an empty list to store the zonal_intensity_deltas
                    zonal_intensity_deltas_rhs = []

                    # Iterate through the list to find zonal_intensity_deltas between consecutive elements
                    for i in range(len(average_intensities_rhs) - 1):
                        diff = round(average_intensities_rhs[i + 1] - average_intensities_rhs[i], 2)
                        zonal_intensity_deltas_rhs.append(diff)

                    self.debug_print(f"Zonal intensity difference between consecutive elements rhs: {zonal_intensity_deltas_rhs} and {len(zonal_intensity_deltas_rhs)}")

                   # Initialize an empty list to store the zonal_intensity_deltas
                    zonal_intensity_deltas_lhs = []

                    # Iterate through the list to find zonal_intensity_deltas between consecutive elements
                    for i in range(len(average_intensities_lhs) - 1):
                        diff = round(average_intensities_lhs[i + 1] - average_intensities_lhs[i], 2)
                        zonal_intensity_deltas_lhs.append(diff)

                    self.debug_print(f"Zonal intensity difference between consecutive elements lhs: {zonal_intensity_deltas_lhs} and {len(zonal_intensity_deltas_lhs)}")

                    null_zone_rhs_possibility = False
                    for diff in zonal_intensity_deltas_rhs:
                        if abs(diff) > self.args['gradientIntensityChange']:
                            null_zone_rhs_possibility = True
                            break  # No need to continue checking if one element meets the condition
                    self.debug_print("Null Zone RHS Possibility:", null_zone_rhs_possibility)

                    null_zone_lhs_possibility = False
                    for diff in zonal_intensity_deltas_lhs:
                        if abs(diff) > self.args['gradientIntensityChange']:
                            null_zone_lhs_possibility = True
                            break  # No need to continue checking if one element meets the condition
                    self.debug_print("Null Zone LHS Possibility:", null_zone_lhs_possibility)

                    #"""
                    deltas=[]
                    #Check if the intensities match
                    for zone in range(num_zones):
                        #print(f"{zone}")
                        if abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]) <= self.args['brightnessTolerance'] and zone > self.args['skipZonesFromCenter']:
                           self.debug_print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
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
                        #pprint.pprint(f"Zone match: {sorted_deltas[0][0]}")

                        line_mark1 = 0 + (int(sorted_deltas[0][0]) * radius // num_zones)
                        #line_mark2 = 0 + ((int(sorted_deltas[0][0])+1) * radius // num_zones)
                        line_mark = int(int(line_mark1) + (int(pixels_per_zone)/2))
                        #draw_symmetrical_line(cropped_image, center_x+line_mark, center_y, 20, color=(255,255,255))
                        #draw_symmetrical_line(cropped_image, center_x-line_mark, center_y, 20, color=(255,255,255))


                        # Define parameters for the arc
                        start_angle = -20
                        end_angle = 20

                        """
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
                        #cv2.waitKey(100)
                        """

                        # Extract zones and deltas for plotting
                        zones = [zone[0] for zone in deltas]
                        differences = [zone[1] for zone in deltas]

                        #============================ Begin:  do same thing on mask ==============================
                        if null_zone_lhs_possibility == True and null_zone_lhs_possibility == True:
                            color = (255, 255, 255)  # White color in BGR
                            self.draw_text(mask_ret, f"NULL Zones found", (center_x-20+top_left_x,center_y+radius-80+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 255, 0), thickness=1)
                        else:
                            color = (0, 0, 255)  # Red color in BGR
                            self.draw_text(mask_ret, f"NULL Zones not found", (center_x-20+top_left_x,center_y+radius-80+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(0, 0, 255), thickness=1)

                        # Use the function to draw the symmetrical arc on the image
                        #self.draw_symmetrical_arc(mask_ret, center_x, center_y, line_mark1, start_angle, end_angle, color)
                        #self.draw_symmetrical_arc(mask_ret, center_x, center_y, line_mark1, start_angle+180, end_angle+180, color)

                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, line_mark1, start_angle, end_angle, color)
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, line_mark1, start_angle+180, end_angle+180, color)
                        
                        zone_pixels = int(pixels_per_zone * int(sorted_deltas[0][0]))
                        zone_inches = float(float(zone_pixels/radius_orig)*self.args['mirrorDiameterInches'])/2
                        self.draw_text(mask_ret, f"Radius: {radius} pixels ", (center_x-20+top_left_x,center_y+radius-20+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Zone: {zone_pixels:.0f} pixels or {zone_inches:.4f} \"", (center_x-20+top_left_x,center_y+radius-40+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Zone match: {sorted_deltas[0][0]} zone of total {num_zones} zones", (center_x-20+top_left_x,center_y+radius-60+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Mirror Diameter: {self.args['mirrorDiameterInches']} \"", (center_x-20+top_left_x,center_y-20+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"Mirror Focal Length: {self.args['mirrorFocalLengthInches']} \"", (center_x-20+top_left_x,center_y-40+top_left_y), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
                        self.draw_text(mask_ret, f"{self.current_timestamp()}", (0,0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, color=(255, 255, 255), thickness=1)
                        self.stale_image = False
                        #============================ End: do same thing on mask ==============================
 
                        # Append to CSV if set
                        if self.args['append_to_csv']:
                            csv_data=[
                                self.current_timestamp(),
                                center_x_orig,
                                center_y_orig,
                                radius_orig,
                                num_zones,
                                int(sorted_deltas[0][0]),
                                zone_pixels,
                                round(zone_inches,3),
                                self.args['mirrorDiameterInches'],
                                self.args['mirrorFocalLengthInches'],
                                self.args['step_size']
                            ]
                            self.write_csv(csv_data)
                            #self.write_csv(','.join(map(str,csv_data)))
                            #csv_line = ','.join(str(item).replace('"', '') for item in csv_data)
                            #self.write_csv(csv_line)

                    else:
                         self.debug_print("No zones have matching intensities!")
                         self.stale_image = True
                    # Check if the image size is smaller than 640x480
                    
                    """
                    if cropped_image.shape[0] < 480 or cropped_image.shape[1] < 640:
                        # Fill the boundary
                        cropped_image = self.fill_image_boundary(cropped_image)
                    """
                    # Return the cropped image
                    # Overlay the mask on the image
                    result = np.copy(image)
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # Convert image to 4 channels (BGR + Alpha)

                    # Blend the mask with the image
                    alpha = 1.0  # Adjust the alpha blending factor (0.0 - fully transparent, 1.0 - fully opaque)
                    cv2.addWeighted(mask_ret, alpha, result, 1.0, 0, result)

                    # Calculate the position to place the text at the center of the image
                    result_center_x = result.shape[1] // 2
                    result_center_y = result.shape[0] // 2
                    # Draw the user text on the result image near center
                    self.draw_text(result, self.args['user_text'], color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, position=(result_center_x-20, result_center_y-20), thickness=1)

                    #cv2.rectangle(result, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color=(0, 255, 0), thickness=2)

                    #Took measurement - Hence save the image
                    if self.args['append_to_csv'] and self.stale_image == False and self.args['enable_disk_rwx_operations']:
                       analyzed_jpg_filename = self.args['csv_filename']+self.current_timestamp()+'.jpg'
                       #analyzed_jpg_file = os.path.join(self.args['folder'], analyzed_jpg_filename)
                       #cv2.imwrite(analyzed_jpg_file, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
                       if platform.system() == "Linux":
                           analyzed_jpg_file = os.path.join(self.args['folder'], analyzed_jpg_filename)
                           if not cv2.imwrite(analyzed_jpg_file, result, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                              raise Exception("Could not write/save image")
                       elif platform.system() == "Windows":
                           save_directory = os.path.join(home_dir, 'FKESAv2Images')
                           os.makedirs(save_directory, exist_ok=True)
                           timestamp = int(time.time())  # Get the current timestamp
                           filename = f"FKESA_v2_{timestamp}.jpg"  # Generate a filename with the timestamp
                           image_path = os.path.join(save_directory,filename)
                           self.debug_print("********************************************************************")
                           self.debug_print(image_path)                           
                           self.debug_print("********************************************************************")
                           #sys.exit(1)
                           #image_path = os.path.join(home_dir, 'Desktop', analyzed_jpg_filename)
                           if not cv2.imwrite(image_path, result):
                              raise Exception("Could not write/save image")

                    #cv2.imwrite("some_file.jpg", cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

                    return cropped_image, result

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
