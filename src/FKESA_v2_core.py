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
from io import BytesIO

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
            'zones': 150,
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
            'enable_disk_rwx_operations': False,
            'step': 0
            # Include default values for other parameters here
        }
        self.stale_image=False

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
                        """
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
                        """
                        writer.writerow([
                                'Diameter Inches',
                                'Focal Length Inches',
                                'Step Size Inches'
                        ])  # Repla
                        writer.writerow([
                                self.args['mirrorDiameterInches'],
                                self.args['mirrorFocalLengthInches'],
                                self.args['step_size']
                        ])  # Replace with your column names

                        writer.writerow([
                            '------ Timestamp ------',
                            #'Mirror X Pixels (Center X) ',
                            #'Mirror Y Pixels (Center Y)',
                            #'Mirror Radius Pixels',
                            'Match Zone number',
                            'Match Zone Pixels',
                            'Match Zone Inches',
                            'Step',
                            'Step Distance Inches'
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

    # data is of format - writer.writerow(['X [pixels], Y [pixels], INTENSITY [0-255]'])
    def write_csv_bottom_flipped_method(self, data):
            csv_filename = self.args['csv_filename']
            csv_file = os.path.join(self.args['folder'], csv_filename)
            is_empty = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0

            # If the file is empty or doesn't exist, write the header
            if is_empty:
                try:
                    with open(csv_file, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                                'Diameter Inches',
                                'Focal Length Inches',
                                'Step Size Inches'
                        ])  # Repla
                        writer.writerow([
                                self.args['mirrorDiameterInches'],
                                self.args['mirrorFocalLengthInches'],
                                self.args['step_size']
                        ])  # Replace with your column names

                        writer.writerow([
                            '------ Timestamp ------',
                            'Match Left Null Zone Distance',
                            'Match Right Null Zone Distance',
                            'Average Match Left/Right Null Zone Distance',
                            'Step',
                            'Step Distance Inches'
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



    def savitzky_golay_coefficients(self, window_size, poly_order):
            """
            Calculate the Savitzky-Golay filter coefficients.
            Parameters:
            window_size (int): The length of the filter window (i.e., the number of coefficients).
                               window_size must be a positive odd integer.
            poly_order (int): The order of the polynomial used to fit the samples.
                              poly_order must be less than window_size.
            Returns:
            numpy.ndarray: The filter coefficients.
            """
            half_window = (window_size - 1) // 2
            # Construct the Vandermonde matrix
            A = np.mat([[k**i for i in range(poly_order + 1)] for k in range(-half_window, half_window + 1)])
            # Compute the inverse of the Gram matrix
            ATA_inv = np.linalg.pinv(A.T * A)
            # Compute the smoothing coefficients
            return np.array(np.dot(ATA_inv, A.T)[0]).flatten()

    def savitzky_golay_smoothing(self, signal, window_size, poly_order):
            """
            Apply Savitzky-Golay polynomial smoothing to a 1D signal.
            Parameters:
            signal (numpy.ndarray): The input signal to smooth.
            window_size (int): The length of the filter window (i.e., the number of coefficients).
                               window_size must be a positive odd integer.
            poly_order (int): The order of the polynomial used to fit the samples.
                              poly_order must be less than window_size.
            Returns:
            numpy.ndarray: The smoothed signal.
            """
            if window_size % 2 == 0:
                raise ValueError("Window size must be an odd integer.")
            if poly_order >= window_size:
                raise ValueError("Polynomial order must be less than window_size.")
            # Get the filter coefficients
            coeffs = self.savitzky_golay_coefficients(window_size, poly_order)
            # Pad the signal at the ends to minimize boundary effects
            half_window = (window_size - 1) // 2
            firstvals = signal[0] - np.abs(signal[1:half_window+1][::-1] - signal[0])
            lastvals = signal[-1] + np.abs(signal[-half_window-1:-1][::-1] - signal[-1])
            padded_signal = np.concatenate((firstvals, signal, lastvals))
            # Apply the convolution
            smoothed_signal = np.convolve(padded_signal, coeffs, mode='valid')
            return smoothed_signal

    def apply_savitzky_golay_to_image(self, image, window_size, poly_order):
            """
            Apply Savitzky-Golay polynomial smoothing to each row of an image.
            Parameters:
            image (numpy.ndarray): The input image to smooth.
            window_size (int): The length of the filter window (i.e., the number of coefficients).
                               window_size must be a positive odd integer.
            poly_order (int): The order of the polynomial used to fit the samples.
                              poly_order must be less than window_size.
            Returns:
            numpy.ndarray: The smoothed image.
            """
            smoothed_image = np.zeros_like(image)
            # Apply the filter to each row
            for i in range(image.shape[0]):
                self.debug_print(f"Smoothing row {i+1}/{image.shape[0]}")
                smoothed_image[i, :] = self.savitzky_golay_smoothing(image[i, :], window_size, poly_order)
            return smoothed_image

    def generate_plot_savitzkly_golay_test(self, plt, pixel_intensity_above, pixel_intensity_below, intersection_points):

        # Plot the pixel intensity vs distance from left to right
        plt.plot(pixel_intensity_above, label='Row above center')
        plt.plot(pixel_intensity_below, label='Row below center')
        plt.xlabel('Distance from left')
        plt.ylabel('Pixel intensity')
        plt.legend()
            
        # Mark intersection points on the plot
        for point in intersection_points:
            plt.axvline(x=point, color='r', linestyle='--')
                       
        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the buffer to a NumPy array
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        # Decode the array to an image using OpenCV
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Convert the image to RGB (OpenCV uses BGR by default)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

        ## Usage:
        ## Convert the image to a format suitable for PySimpleGUI
        #img_bytes = cv2.imencode('.png', img)[1].tobytes()
        #
        # Create a PySimpleGUI window to display the image
        #layout = [[sg.Image(data=img_bytes, key='-IMAGE-')]]
        #window = sg.Window('Plot', layout, finalize=True)
        ## Display the window
        #while True:
        #    event, values = window.read()
        #    if event == sg.WINDOW_CLOSED:
        #        break 
        #
        #window.close()

    def build_savitzky_golan_flip_test(self, image):
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
                    height, width, channels = cropped_image.shape


                    # Split the image into two halves
                    top_half = cropped_image[:height//2, :]
                    bottom_half = cropped_image[height//2:, :]

                    # Flip the bottom half
                    flipped_bottom_half = cv2.flip(bottom_half, 1)

                    # Combine the top half and flipped bottom half
                    flipped_image = np.concatenate([top_half, flipped_bottom_half], axis=0)

                    # Convert the flipped image to grayscale
                    flipped_image_gray = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2GRAY)

                    #cv2.imwrite('flipped_gray.jpg', flipped_image_gray, [cv2.IMWRITE_JPEG_QUALITY, 100])


                    flipped_image_gray = self.apply_savitzky_golay_to_image(flipped_image_gray, 11,2)

                    # Get the image shape
                    flipped_height, flipped_width = flipped_image_gray.shape

                    # Extract the rows of interest
                    row_above = flipped_image_gray[flipped_height//2-1, :]
                    row_below = flipped_image_gray[flipped_height//2+1, :]

                    # Calculate the pixel intensity for each column in the rows of interest
                    pixel_intensity_above = row_above
                    pixel_intensity_below = row_below

                    # Find intersection points
                    intersection_points = []
                    for x in range(flipped_width):
                            if pixel_intensity_above[x] == pixel_intensity_below[x]:
                                intersection_points.append(x)
                            else:
                                # Check if the pixel intensities cross between adjacent columns
                                if x > 0:
                                    if (pixel_intensity_above[x-1] < pixel_intensity_below[x-1] and pixel_intensity_above[x] > pixel_intensity_below[x]) or \
                                       (pixel_intensity_above[x-1] > pixel_intensity_below[x-1] and pixel_intensity_above[x] < pixel_intensity_below[x]):
                                        intersection_points.append(x)

                    # Print intersection points
                    print(f'Intersection points: {intersection_points}')
                    print("=========")

                    # Get image dimensions
                    height, width = cropped_image.shape[:2]

                    # Define the center of the image
                    center_x = width // 2
                    center_y = height // 2

                    # Create a blank mask
                    mask = np.zeros((height, width), dtype=np.uint8)

                    # Mark the intersection points with white lines on the original image
                    marked_image = image.copy()
                    line_length = 40  # length of the line
                    half_line_length = line_length // 2

                    # Flag to select closest or farthest points
                    #select_farthest = True  # Set this to False to select closest points
                    select_farthest = False  # Set this to False to select closest points

                    # TODO (Pratik) make a slider or input text box to adjust this value.
                    #skip_range = 20

                    # Calculate skip_range based on the ratio of mirrorDiameterInches to mirrorFocalLengthInches
                    f_ratio = float(float(self.args['mirrorDiameterInches'])/float(self.args['mirrorFocalLengthInches']))
                    print(f"f_ratio is {f_ratio}")
                    print(f"width is {width}")
                    #Some placeholder
                    #skip_range = int(float(float(self.args['mirrorDiameterInches']) * float(f_ratio) * float(width//2)))
                    skip_range = int(float(float(f_ratio) * float(width//2)))
                    print(f"skip_range is {skip_range}")
             

                    # Skip pixels towards the edge too.(Alan's observation)
                    edge_margin = 1

                    filtered_points = [
                       point for point in intersection_points
                       if abs(point - center_x) > skip_range and point > edge_margin and point < (width - edge_margin)
                    ]

                    # Filter out points within the skipped range from center (old logic)
                    #filtered_points = [point for point in intersection_points if abs(point - center_x) > skip_range]

                    # Separate points to the left and right of the center
                    left_points = [point for point in filtered_points if point < center_x]
                    right_points = [point for point in filtered_points if point > center_x]

                    # Calculate the closest or farthest points based on the flag
                    if select_farthest:
                            farthest_left_point = max(left_points, default=None, key=lambda x: abs(x - center_x))
                            farthest_right_point = max(right_points, default=None, key=lambda x: abs(x - center_x))
                            farthest_left_distance = abs(farthest_left_point - center_x) if farthest_left_point is not None else None
                            farthest_right_distance = abs(farthest_right_point - center_x) if farthest_right_point is not None else None
                            print(f'Farthest left point: {farthest_left_point}, Distance: {farthest_left_distance}')
                            print(f'Farthest right point: {farthest_right_point}, Distance: {farthest_right_distance}')
                    else:
                            closest_left_point = min(left_points, default=None, key=lambda x: abs(x - center_x))
                            closest_right_point = min(right_points, default=None, key=lambda x: abs(x - center_x))
                            closest_left_distance = abs(closest_left_point - center_x) if closest_left_point is not None else None
                            closest_right_distance = abs(closest_right_point - center_x) if closest_right_point is not None else None
                            print(f'Closest left point: {closest_left_point}, Distance: {closest_left_distance}')
                            print(f'Closest right point: {closest_right_point}, Distance: {closest_right_distance}')

                    # Use the selected distances for further calculations
                    if select_farthest:
                       left_distance = farthest_left_distance
                       right_distance = farthest_right_distance
                    else:
                       left_distance = closest_left_distance
                       right_distance = closest_right_distance

                    if left_distance is not None and right_distance is not None:
                       left_null_zone = round(float(float(left_distance / radius_orig) * self.args['mirrorDiameterInches']) / 2, 2)
                       right_null_zone = round(float(float(right_distance / radius_orig) * self.args['mirrorDiameterInches']) / 2, 2)

                       print(f'null zone left point: {left_null_zone}')
                       print(f'null zone right point: {right_null_zone}')
                       # Calculate the average of left and right null zones
                       average_null_zone = round((left_null_zone + right_null_zone) / 2, 2)
                       print(f'Average null zone: {average_null_zone}')

                    for point in filtered_points:
                        start_point = (point + top_left_x, height // 2 + top_left_y - half_line_length)
                        end_point = (point + top_left_x, height // 2 + top_left_y + half_line_length)
                        cv2.line(marked_image, start_point, end_point, (255, 0, 0), 1)

                    # Append to CSV if set
                    if self.args['append_to_csv'] and left_distance is not None and right_distance is not None:
                            csv_data = [
                                self.current_timestamp(),
                                left_null_zone,
                                right_null_zone,
                                average_null_zone,  # Add the average null zone to the CSV data
                                self.args['step'],
                                float(float(self.args['step_size']) * int(self.args['step']))
                            ]
                            self.write_csv_bottom_flipped_method(csv_data)


                    # One may uncomment to save plot image but it does have a significant impact on the processing speed
                    #plot_image = self.generate_plot_savitzkly_golay_test(plt, pixel_intensity_above, pixel_intensity_below, filtered_points)
            
                    """
                    # Define parameters for the arc
                    start_angle = -20
                    end_angle = 20
                    color = (255, 255, 255)  # White color in BGR

                    for point in intersection_points:
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, point, start_angle, end_angle, color)
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, point, start_angle+180, end_angle+180, color)

                        point_inches = float(float(point/radius_orig)*self.args['mirrorDiameterInches'])/2
                        # Append to CSV if set
                        if self.args['append_to_csv']:
                            csv_data=[
                                self.current_timestamp(),
                                #center_x_orig,
                                #center_y_orig,
                                #radius_orig,
                                int(point),
                                point,
                                round(point_inches,3),
                                self.args['step'],
                                float(float(self.args['step_size'])*int(self.args['step']))
                            ]
                            self.write_csv(csv_data)
                            #self.write_csv(','.join(map(str,csv_data)))
                            #csv_line = ','.join(str(item).replace('"', '') for item in csv_data)
                            #self.write_csv(csv_line)

                    """                   
                    # Return the cropped image
                    # Overlay the mask on the image
                    result = np.copy(marked_image)
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)  # Convert image to 4 channels (BGR + Alpha)

                    # Blend the mask with the image
                    alpha = 1.0  # Adjust the alpha blending factor (0.0 - fully transparent, 1.0 - fully opaque)
                    cv2.addWeighted(mask_ret, alpha, result, 1.0, 0, result)

                    # Calculate the position to place the text at the center of the image
                    result_center_x = result.shape[1] // 2
                    result_center_y = result.shape[0] // 2
                    # Draw the user text on the result image near center
                    self.draw_text(result, self.args['user_text'], color=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, position=(result_center_x-radius+20, result_center_y-radius+20), thickness=2)

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





    def build_manual_test(self, image):
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

                    #Took measurement - Hence save the image
                    if self.args['append_to_csv'] and self.stale_image == False and self.args['enable_disk_rwx_operations']:
                       phi_final_image_filename = 'PHI_'+self.args['csv_filename']+self.current_timestamp()+'.jpg'
                       #phi_final_image_file = os.path.join(self.args['folder'], phi_final_image_filename)
                       #cv2.imwrite(phi_final_image_file, phi_final_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
                       if platform.system() == "Linux":
                           phi_final_image_file = os.path.join(self.args['folder'], phi_final_image_filename)
                           if not cv2.imwrite(phi_final_image_file, phi_final_image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                              raise Exception("Could not write/save image")
                       elif platform.system() == "Windows":
                           save_directory = os.path.join(home_dir, 'FKESAv2Images')
                           os.makedirs(save_directory, exist_ok=True)
                           timestamp = int(time.time())  # Get the current timestamp
                           filename = f"PHI_FKESA_v2_{timestamp}.jpg"  # Generate a filename with the timestamp
                           image_path = os.path.join(save_directory,filename)
                           self.debug_print("********************************************************************")
                           self.debug_print(image_path)
                           self.debug_print("********************************************************************")
                           #sys.exit(1)
                           #image_path = os.path.join(home_dir, 'Desktop', phi_final_image_filename)
                           if not cv2.imwrite(image_path, phi_final_image):
                              raise Exception("Could not write/save phi image")

                    # Get image dimensions
                    height, width = cropped_image.shape[:2]

                    # Define the center of the image
                    center_x = width // 2
                    center_y = height // 2

                    # Define the number of zones
                    num_zones = self.args['zones']

                    if num_zones > 150:
                        self.debug_print("WARNING!!! - Number of zones exceed 150. Limiting to 150.")
                        num_zones = 150
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


                    deltas=[]
                    #Check if the intensities match
                    for zone in range(num_zones):
                      if abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]) <= self.args['brightnessTolerance'] and zone > self.args['skipZonesFromCenter']:
                        self.debug_print(f"{zone} intensity matches with difference {abs(average_intensities_rhs[zone] - average_intensities_lhs[zone]):.4f}" )
                        difference = abs(average_intensities_rhs[zone] - average_intensities_lhs[zone])
                        deltas.append((zone,difference))
                        #print(f"Zone is {zone}")

                    # Define parameters for the arc
                    start_angle = -20
                    end_angle = 20
                    color = (255, 255, 255)  # White color in BGR

                    if deltas:
                       for zone_index, _ in deltas:
                        line_mark1 = 0 + (int(zone_index) * radius // num_zones)
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, line_mark1, start_angle, end_angle, color)
                        self.draw_symmetrical_arc(mask_ret, center_x+top_left_x, center_y+top_left_y, line_mark1, start_angle+180, end_angle+180, color)

                        zone_pixels = int(pixels_per_zone * zone_index)
                        zone_inches = float(float(zone_pixels/radius_orig)*self.args['mirrorDiameterInches'])/2
                        #"""
                        # Append to CSV if set
                        if self.args['append_to_csv']:
                            csv_data=[
                                self.current_timestamp(),
                                #center_x_orig,
                                #center_y_orig,
                                #radius_orig,
                                int(zone_index),
                                zone_pixels,
                                round(zone_inches,3),
                                self.args['step'],
                                float(float(self.args['step_size'])*int(self.args['step']))
                            ]
                            self.write_csv(csv_data)
                            #self.write_csv(','.join(map(str,csv_data)))
                            #csv_line = ','.join(str(item).replace('"', '') for item in csv_data)
                            #self.write_csv(csv_line)
                        #"""

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
                    self.draw_text(result, self.args['user_text'], color=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, position=(result_center_x-radius+20, result_center_y-radius+20), thickness=2)

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

                    if num_zones > 150:
                        self.debug_print("WARNING!!! - Number of zones exceed 150. Limiting to 150.")
                        num_zones = 150
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
                            """
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
                            """
                            csv_data=[
                                self.current_timestamp(),
                                #center_x_orig,
                                #center_y_orig,
                                #radius_orig,
                                int(sorted_deltas[0][0]),
                                zone_pixels,
                                round(zone_inches,3),
                                self.args['step'],
                                float(float(self.args['step_size'])*int(self.args['step']))
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
                    self.draw_text(result, self.args['user_text'], color=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, position=(result_center_x-radius+20, result_center_y-radius+20), thickness=2)

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

                    if num_zones > 150:
                        self.debug_print("WARNING!!! - Number of zones exceed 150. Limiting to 150.")
                        num_zones = 150
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
                            """
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
                            """
                            csv_data=[
                                self.current_timestamp(),
                                #center_x_orig,
                                #center_y_orig,
                                #radius_orig,
                                int(sorted_deltas[0][0]),
                                zone_pixels,
                                round(zone_inches,3),
                                self.args['step'],
                                float(float(self.args['step_size'])*int(self.args['step']))
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
                    self.draw_text(result, self.args['user_text'], color=(0, 0, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, position=(result_center_x-radius+20, result_center_y-radius+20), thickness=2)

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
