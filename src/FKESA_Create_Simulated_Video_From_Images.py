#!/usr/bin/env python3

#Usage: ./create_movie.py resized_images output_video.avi --fps 30

import cv2
import argparse
import os

def main(input_folder_path, output_video_path, fps, duration):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(input_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        raise ValueError("No images found in the specified folder.")

    # Sort the image files to maintain order
    image_files.sort()

    # Print the list of image files found
    print(f"Found {len(image_files)} images in the folder:")
    for image_file in image_files:
        print(f" - {image_file}")

    # Read the first image to get dimensions
    first_image_path = os.path.join(input_folder_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise ValueError(f"Unable to read the first image: {first_image_path}")

    # Resize the first image to 640x480 to get the dimensions
    resized_first_image = cv2.resize(first_image, (640, 480))
    height, width, layers = resized_first_image.shape
    print(f"Resized image dimensions: {width}x{height}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can use other codecs like 'MJPG'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Calculate the number of frames per image for the specified duration
    frames_per_image = fps * duration

    # Write each image to the video file for the calculated number of frames
    for image_file in image_files:
        image_path = os.path.join(input_folder_path, image_file)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read the image: {image_path}")

        # Resize the image to 640x480
        resized_image = cv2.resize(image, (640, 480))

        print(f"Writing {image_file} to the video...")
        for _ in range(frames_per_image):
            out.write(resized_image)

    # Release the VideoWriter
    out.release()

    print(f"Video created successfully at {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from a folder of images.")
    parser.add_argument("input_folder_path", help="Path to the folder containing images")
    parser.add_argument("output_video_path", help="Path to the output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds for each image (default: 60)")

    args = parser.parse_args()

    main(args.input_folder_path, args.output_video_path, args.fps, args.duration)

