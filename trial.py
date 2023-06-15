import cv2
import numpy as np

def dehaze_image(image, omega=0.95, tmin=0.1):
    """Dehazes an image using Dark Channel Prior and adaptive histogram equalization.

    Args:
        image: The input image.
        omega: Parameter controlling the amount of haze removal (default: 0.95).
        tmin: Minimum transmission value to avoid over-darkening (default: 0.1).

    Returns:
        The dehazed image.
    """

    # Calculate the dark channel of the image
    min_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((15, 15), np.uint8))

    # Estimate the atmospheric light
    top_percentile = int(dark_channel.size * 0.001)
    top_percentile = max(0, min(top_percentile, 100))  # Ensure it is within [0, 100]
    atmospheric_light = np.percentile(dark_channel, top_percentile)

    # Estimate the transmission map
    transmission = 1 - omega * dark_channel / atmospheric_light
    transmission = np.maximum(transmission, tmin)

    # Perform adaptive histogram equalization on each channel
    dehazed_image = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        dehazed_image[:, :, channel] = cv2.equalizeHist(
            (image[:, :, channel].astype(np.float32) / transmission).astype(np.uint8)
        )

    # Clip the pixel values and convert back to uint8
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    return dehazed_image

def main():
    # Load the input image
    image = cv2.imread(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg')

    # Dehaze the image
    dehazed_image = dehaze_image(image)

    # Save the dehazed image
    cv2.imwrite(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\dehaze.jpeg', dehazed_image)

if __name__ == '__main__':
    main()
