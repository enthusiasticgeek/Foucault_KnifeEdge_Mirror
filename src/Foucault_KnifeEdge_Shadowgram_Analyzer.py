#!/usr/bin/env python3
# ======================================================
# Author: Pratik M Tambe <enthusiasticgeek@gmail.com>
# Date: November 18, 2023
# Foucault Knife Edge Detection Test on Mirror 
# Target: National Capital Astronomers (NCA), Washington DC, USA
# Program: Amateur Telescope Making (ATM) Workshop
# ======================================================
import cv2
import numpy as np
import argparse
import math
import random
import signal
import sys
import csv

# Refrain from using larger images
# Smaller photos like 640x480 is ideal for faster processing and better results

# Helper functions
def signal_handler(sig, frame):
    # Handle Ctrl+C (SIGINT)
    print("\nOperation interrupted by user.")
    sys.exit(0)

def resize_image(image, max_width=640):
    height, width, _ = image.shape
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        return cv2.resize(image, (max_width, new_height))
    return image

def print_intensity_along_line(lst, image, start_point, end_point):
    # Ensure start_point is the leftmost point
    if start_point[0] > end_point[0]:
        start_point, end_point = end_point, start_point

    for x_coord in range(start_point[0], end_point[0] + 1):
        y_coord = start_point[1] + (x_coord - start_point[0]) * (end_point[1] - start_point[1]) // (end_point[0] - start_point[0])
        
        # Ensure y_coord is within image bounds
        if 0 <= x_coord < image.shape[1] and 0 <= y_coord < image.shape[0]:
            intensity = image[y_coord, x_coord]
            print(f"Intensity at ({x_coord}, {y_coord}): {intensity}")
            lst.append((x_coord,y_coord,intensity))
    return lst

def get_average_intensity(image, x, y):
    # Calculate the 5x5 pixel neighborhood
    neighborhood = image[y - 2:y + 3, x - 2:x + 3]
    # Calculate the average intensity
    average_intensity = neighborhood.mean()
    return average_intensity


def write_matching_intensities_to_csvi(matches):
    with open('matching_intensities.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Less Than X1 (x, y, intensity, distance from X1)', 'Greater Than X1 (x, y, intensity, distance from X1)'])

        for match in matches:
            lt_point = match[0]
            gt_point = match[1]
            writer.writerow([lt_point, gt_point])

def write_matching_intensities_to_csv(matches):
    with open('matching_intensities.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Less Than X1 (x, y, intensity, distance from X1)', 'Greater Than X1 (x, y, intensity, distance from X1)'])

        for match in matches:
            lt_point = ', '.join(map(str, match[0]))  # Convert tuple to string and remove brackets
            gt_point = ', '.join(map(str, match[1]))  # Convert tuple to string and remove brackets
            writer.writerow([lt_point, gt_point])

def draw_symmetrical_line(image, x, y, line_length):
    cv2.line(image, (x, y - line_length), (x, y + line_length), (255, 255, 255), thickness=1)

def find_matching_intensities_and_draw_lines(lst, x1, tolerance, image, line_length):
    matches = []
    less_than_x1 = []
    greater_than_x1 = []

    for point in lst:
        x, y, intensity = point
        distance = abs(x - x1)  # Calculate distance along x-axis

        if x < x1:
            less_than_x1.append((x, y, intensity, distance))
        elif x > x1:
            greater_than_x1.append((x, y, intensity, distance))

    # Compare average intensities within the same distance in both segments
    for lt_point in less_than_x1:
        for gt_point in greater_than_x1:
            if lt_point[3] == gt_point[3]:
                lt_average = get_average_intensity(image, lt_point[0], lt_point[1])
                gt_average = get_average_intensity(image, gt_point[0], gt_point[1])

                if abs(lt_average - gt_average) <= tolerance:
                    print(f"Matching intensities found:")
                    print(f"Less than x1: {lt_point}")
                    print(f"Greater than x1: {gt_point}")
                    print("-----")

                    matches.append((lt_point, gt_point))

                    # Draw symmetrical lines centered at the x-coordinate
                    draw_symmetrical_line(image, lt_point[0], lt_point[1], line_length)
                    draw_symmetrical_line(image, gt_point[0], gt_point[1], line_length)
    # Collect data in CSV
    write_matching_intensities_to_csv(matches)

def main():
    signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler

    try:
        parser = argparse.ArgumentParser(description='Detect largest circle in an image')
        parser.add_argument('filename', help='Path to the image file')
        parser.add_argument('--minDist', type=int, default=50, help='Minimum distance between detected circles')
        parser.add_argument('--param1', type=int, default=55, help='First method-specific parameter')
        parser.add_argument('--param2', type=int, default=60, help='Second method-specific parameter')
        parser.add_argument('--minRadius', type=int, default=10, help='Minimum circle radius')
        parser.add_argument('--maxRadius', type=int, default=0, help='Maximum circle radius')
        parser.add_argument('--canTolerance', type=int, default=2, help='Candidates y-axis tolerance')
        parser.add_argument('--drawContours', type=int, default=0, help='Draw contours')
        parser.add_argument('--drawNestedContours', type=int, default=0, help='Draw Nested contours')
        parser.add_argument('--drawCircles', type=int, default=1, help='Draw mirror circle(s)')
        parser.add_argument('--brightnessTolerance', type=int, default=30, help='Brightness tolerance value for two contour regions to be considered as similar brightness')
        parser.add_argument('--displayWindowPeriod', type=int, default=10000, help='Display window period 10 seconds. Set to 0 for infinite window period.')
        args = parser.parse_args()

        try:
                # Load the image and resize it
                image = cv2.imread(args.filename)
                if image is None:
                    raise FileNotFoundError("Image file not found or cannot be read.")

                image = resize_image(image)

                if image is None:
                    print(f"Unable to load the image from {args.filename}")
                    return

                # Convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)

                # Use thresholding to create a binary image
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Use adaptive thresholding to create a binary image
                #thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)

                # Find contours and hierarchy
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if args.drawNestedContours == 1: 
                        # Identify and draw only nested contours
                        for i in range(len(contours)):
                            # Check if the contour has a parent (nested contour)
                            if hierarchy[0][i][3] != -1:
                                cv2.drawContours(image, [contours[i]], -1, (0, 0, 255), 2)  # Draw nested contours in red

                # Find contours in the binary image
                #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Sort contours based on their area in descending order
                contours = sorted(contours, key=cv2.contourArea, reverse=True)

                # Draw the contours on the original image
                result = image.copy()

                if args.drawContours == 1:
                    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

                # Apply Hough Circle Transform with user-defined parameters
                circles = cv2.HoughCircles(
                    blurred,
                    cv2.HOUGH_GRADIENT, dp=1, minDist=args.minDist, param1=args.param1, param2=args.param2,
                    minRadius=args.minRadius, maxRadius=args.maxRadius
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    largest_circle = circles[0, 0]
                    for i in circles[0, :]:
                        if i[2] > largest_circle[2]:
                            largest_circle = i
                    x, y, r = largest_circle
                    cv2.circle(result, (x, y), 5, (0, 0, 255), -1)
                    if args.drawCircles == 1:
                            cv2.circle(result, (x, y), r, (0, 0, 255), 2)
                            # Draw red vertical line inside the circle
                            # cv2.line(result, (x, y - r), (x, y + r), (0, 0, 255), 2)
                            # Get intensity at the center of the circle

                            # Iterate through each contour to find the one containing the desired y-coordinate
                            lst=[]
                            for contour in contours:
                                print("===== New contour =====")
                                # Find the bounding rectangle of the contour
                                x1, y1, w1, h1 = cv2.boundingRect(contour)
                            
                                # Check if the desired y-coordinate is within this contour
                                if y1 < y < y1 + h1:
                                   # Draw a line constrained within the bounds of this contour
                                   cv2.line(result, (x1, y), (x1 + w1, y), (255, 0, 0), 2)  # Blue line along x-axis
                                   print_intensity_along_line(lst, gray, (x1,y), (x1+w1,y))
                                print("=======================")
                            print(lst)
                            find_matching_intensities_and_draw_lines(lst,x,args.brightnessTolerance,gray,10)

                if args.drawContours == 1:
                   cv2.imshow('Image with Segmentation Boundaries and Circle/ Contours on Shadowgram', result)
                #cv2.imshow('Threshold', thresh)
                cv2.imshow('Image with markers on Shadowgram', gray)
                cv2.waitKey(args.displayWindowPeriod) # Wait 10 seconds max. Set to 0 for infinite
                cv2.destroyAllWindows()
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")

if __name__ == '__main__':
    main()

