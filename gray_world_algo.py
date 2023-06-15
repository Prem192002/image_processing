import cv2
import numpy as np

def underwater_image_enhancement(image):
    """Enhances an underwater image to remove water and restore original colors.

    Args:
        image: The input underwater image.

    Returns:
        The enhanced image with removed water and restored colors.
    """

    # Convert image to float32 format
    image = image.astype(np.float32) / 255.0

    # Compute the average values for each color channel
    avg_channel_values = np.mean(image, axis=(0, 1))

    # Compute the correction factors for each color channel
    correction_factors = np.divide(avg_channel_values, np.max(avg_channel_values))

    # Apply the correction factors to each pixel
    enhanced_image = np.divide(image, correction_factors)

    # Clip the pixel values to the valid range [0, 1] and convert back to uint8
    enhanced_image = np.clip(enhanced_image, 0, 1.0)
    enhanced_image = (enhanced_image * 255).astype(np.uint8)

    return enhanced_image

def main():
    # Load the input image
    image = cv2.imread(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg')

    # Enhance the underwater image
    enhanced_image = underwater_image_enhancement(image)

    # Save the enhanced image
    cv2.imwrite(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\dehaze.jpeg', enhanced_image)

if __name__ == '__main__':
    main()
