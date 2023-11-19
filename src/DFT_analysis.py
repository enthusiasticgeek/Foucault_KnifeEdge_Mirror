#!/usr/bin/env python3

import cv2
import numpy as np
import argparse
from datetime import datetime

def reconstruct_image(image_path):
    try:
        # Read the image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply DFT to the image
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Compute magnitude spectrum (optional for visualization)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        # Reconstruct the image using inverse DFT
        fshift = np.fft.ifftshift(dft_shift)
        reconstructed_image_complex = cv2.idft(fshift)
        reconstructed_image = cv2.magnitude(reconstructed_image_complex[:, :, 0], reconstructed_image_complex[:, :, 1])

        # Normalize and convert the reconstructed image to uint8
        reconstructed_image = cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display the reconstructed image and magnitude spectrum
        cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
        cv2.imshow('Reconstructed Image', reconstructed_image)

        # Get current timestamp
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")

        # Formulate the filename with timestamp
        filename = f"magnitude_spectrum_{current_time}.png"

        # Save the magnitude spectrum image
        cv2.imwrite(filename, magnitude_spectrum)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    except cv2.error as e:
        print(f"OpenCV error occurred: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Reconstruct image using DFT and display magnitude spectrum')
    parser.add_argument('image', help='Path to the input image file')
    args = parser.parse_args()

    # Reconstruct and display the image
    reconstruct_image(args.image)

