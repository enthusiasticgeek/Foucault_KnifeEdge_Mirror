#!/usr/bin/env python3

import cv2
import argparse

def main(input_image_path, output_video_path, fps, frame_count):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        raise ValueError("Image not found or unable to read the image.")

    # Get the dimensions of the image
    height, width, layers = image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # You can use other codecs like 'MJPG'

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write the image to the video file repeatedly
    for _ in range(frame_count):
        out.write(image)

    # Release the VideoWriter
    out.release()

    print(f"Video created successfully at {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from an image.")
    parser.add_argument("input_image_path", help="Path to the input image")
    parser.add_argument("output_video_path", help="Path to the output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--frame_count", type=int, default=3000, help="Number of frames to repeat the image (default: 3000)")

    args = parser.parse_args()

    main(args.input_image_path, args.output_video_path, args.fps, args.frame_count)

