#!/usr/bin/env python3
# ======================================================
# Author: Pratik M Tambe <enthusiasticgeek@gmail.com>
# Date: November 18, 2023
# Foucault Knife Edge Detection Test on Mirror 
# Target: National Capital Astronomers (NCA), Washington DC, USA
# Program: Amateur Telescope Making (ATM) Workshop
# ======================================================
import matplotlib
# On some Linux systems may need to uncomment this.
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt

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

def print_intensity_along_line_with_threshold(lst, image, start_point, end_point, distance_threshold):
    # Ensure start_point is the leftmost point
    if start_point[0] > end_point[0]:
        start_point, end_point = end_point, start_point

    for x_coord in range(start_point[0], end_point[0] + 1):
        y_coord = start_point[1] + (x_coord - start_point[0]) * (end_point[1] - start_point[1]) // (end_point[0] - start_point[0])
        
        # Ensure y_coord is within image bounds
        if 0 <= x_coord < image.shape[1] and 0 <= y_coord < image.shape[0]:
            distance_from_center = np.sqrt((x_coord - image.shape[1] // 2) ** 2 + (y_coord - image.shape[0] // 2) ** 2)
            if distance_from_center > distance_threshold:
                intensity = image[y_coord, x_coord]
                print(f"Intensity at ({x_coord}, {y_coord}): {intensity}")
                lst.append((x_coord, y_coord, intensity))
    return lst


def get_average_intensity(image, x, y):
    # Calculate the 5x5 pixel neighborhood
    neighborhood = image[y - 2:y + 3, x - 2:x + 3]
    # Calculate the average intensity
    average_intensity = neighborhood.mean()
    return average_intensity


def write_matching_intensities_to_csv(matches, save_plot, plot_output):
    with open(plot_output+'.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Less Than X1 (x, y, intensity, distance from X1)', 'Greater Than X1 (x, y, intensity, distance from X1)'])

        for match in matches:
            lt_point = ', '.join(map(str, match[0]))  # Convert tuple to string and remove brackets
            gt_point = ', '.join(map(str, match[1]))  # Convert tuple to string and remove brackets
            writer.writerow([lt_point, gt_point])

    plt.figure(figsize=(12,8)) # 12 inches x 8 inches
    # Read the CSV to plot the points
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # List of colors for different segments
    with open(plot_output+'.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row

        for i, row in enumerate(reader):
            lt_point = list(map(int, row[0].split(', ')))  # Convert string back to tuple
            gt_point = list(map(int, row[1].split(', ')))  # Convert string back to tuple

            # Plot the points with intensities on Y-axis
            plt.scatter(i, lt_point[2], color=colors[i % len(colors)], label=f'Point {i}')
            plt.scatter(i, gt_point[2], color=colors[i % len(colors)])

            # Annotate 'lt' points above and 'gt' points below the plotted points
            plt.annotate(f'({lt_point[0]}, {lt_point[1]} [{lt_point[2]}])', (i, lt_point[2]), textcoords="offset points", xytext=(0,8), ha='center', va='bottom')
            plt.annotate(f'({gt_point[0]}, {gt_point[1]} [{gt_point[2]}])', (i, gt_point[2]), textcoords="offset points", xytext=(0,-8), ha='center', va='top')

            # Calculate and plot the absolute difference between 'lt' and 'gt' intensities as separate points
            abs_diff = abs(lt_point[2] - gt_point[2])
            plt.scatter(i, abs_diff, color='black', marker='x', label=f'Abs Diff {i}')

            # Annotate the absolute difference values beside the 'x' points
            plt.text(i + 0.2, abs_diff, f'{abs_diff}', fontsize=8, color='black', ha='left', va='center')

    # Set plot labels and title
    plt.xlabel('Segment Index [Rows in CSV file]')
    plt.ylabel('Intensity')
    plt.title('Matching Intensities')
    # Show the plot legend
    plt.legend()
    if save_plot == 1:
       # Save the plot as an image (e.g., PNG, PDF, SVG, etc.)
       plt.savefig(plot_output + ".plot.png")
    # Show the plot
    plt.show()

def draw_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(255, 255, 255), thickness=1):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x, text_y = position
    text_y = max(text_y, text_size[1])  # Ensure the text doesn't go out of the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def draw_symmetrical_line(image, x, y, line_length, color):
    cv2.line(image, (x, y - line_length), (x, y + line_length), color, thickness=1)

def find_matching_intensities_and_draw_lines(lst, x1, y1, r1, tolerance, image, line_length, save_plot, plot_output):
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
                    draw_symmetrical_line(image, lt_point[0], lt_point[1], line_length, (255,255,255))
                    draw_symmetrical_line(image, gt_point[0], gt_point[1], line_length, (255,255,255))

    # Draw the center of the mirror
    draw_text(image, f"({x1},{y1})", (x1-20,y1-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
    draw_text(image, f"Radius: {r1}", (x1-20,y1+r1-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)

    # Find the closest match and draw it!
    for i in matches:
        print(f"{i[0]},{i[1]}")
        if abs(int(i[0][2])-int(i[1][2])) < 1:
           draw_text(image, f"({i[0][0]},{i[0][1]})", (i[0][0]-20,i[0][1]-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
           draw_text(image, f"Intensity: {i[0][2]}", (i[0][0]-30,i[0][1]+20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
           draw_symmetrical_line(image, i[0][0],i[0][1], line_length+10, color=(255,255,255))
           draw_text(image, f"({i[1][0]},{i[1][1]})", (i[1][0]-20,i[1][1]-20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
           draw_text(image, f"Intensity: {i[1][2]}", (i[1][0]-30,i[1][1]+20), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, color=(255, 255, 255), thickness=1)
           draw_symmetrical_line(image, i[1][0], i[1][1], line_length+10, color=(255,255,255))


    # Collect data in CSV
    write_matching_intensities_to_csv(matches, save_plot, plot_output)

def main():
    signal.signal(signal.SIGINT, signal_handler)  # Register the signal handler

    try:
        parser = argparse.ArgumentParser(description='Detect largest circle in an image')
        parser.add_argument('filename', help='Path to the image file')
        parser.add_argument('-d', '--minDist', type=int, default=50, help='Minimum distance between detected circles')
        parser.add_argument('-p1', '--param1', type=int, default=55, help='First method-specific parameter')
        parser.add_argument('-p2', '--param2', type=int, default=60, help='Second method-specific parameter')
        parser.add_argument('-minR', '--minRadius', type=int, default=10, help='Minimum circle radius')
        parser.add_argument('-maxR', '--maxRadius', type=int, default=0, help='Maximum circle radius')
        parser.add_argument('-dc', '--drawContours', type=int, default=0, help='Draw contours')
        parser.add_argument('-dnc', '--drawNestedContours', type=int, default=0, help='Draw Nested contours')
        parser.add_argument('-dr', '--drawCircles', type=int, default=1, help='Draw mirror circle(s)')
        parser.add_argument('-bt', '--brightnessTolerance', type=int, default=20, help='Brightness tolerance value for intensity calculation. Default value is 20')
        parser.add_argument('-dwp', '--displayWindowPeriod', type=int, default=10000, help='Display window period 10 seconds. Set to 0 for infinite window period.')
        parser.add_argument('-spnc', '--skipPixelsNearCenter', type=int, default=40, help='Skip the pixels that are too close to the center of the mirror for intensity calculation. Default value is 40')
        parser.add_argument('-svi', '--saveImage', type=int, default=1, help='Save the Analysis Image on the disk with the timestamp (value changed to 1). Default value is 1')
        parser.add_argument('-svp', '--savePlot', type=int, default=1, help='Save the Analysis Plot on the disk with the timestamp (value changed to 1). Default value is 1')
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
                                   #print_intensity_along_line(lst, gray, (x1,y), (x1+w1,y))
                                   print_intensity_along_line_with_threshold(lst, gray, (x1,y), (x1+w1,y),args.skipPixelsNearCenter)
                                print("=======================")
                            print(lst)
                            find_matching_intensities_and_draw_lines(lst,x,y,r,args.brightnessTolerance,gray,2,args.savePlot,args.filename)

                if args.drawContours == 1:
                   cv2.imshow('Image with Segmentation Boundaries and Circle/ Contours on Shadowgram', result)
                if args.saveImage == 1:
                   cv2.imwrite(args.filename + '.analysis.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 100])
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

