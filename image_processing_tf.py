import cv2
import numpy as np

def dehaze_adaptive_histogram_equalization(image):
  """Dehazes an image using adaptive histogram equalization.

  Args:
    image: The input image.

  Returns:
    The dehazed image.
  """

  # Equalize the histogram of each channel of the image.
  for channel in range(3):
    image[:, :, channel] = cv2.equalizeHist(image[:, :, channel])

  return image

def main():
  # Load the input image.
  image = cv2.imread(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg')

  # Dehaze the image.
  dehazed_image = dehaze_adaptive_histogram_equalization(image)

  # Save the dehazed image.
  cv2.imwrite(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\imag20.jpeg', dehazed_image)

if __name__ == '__main__':
  main()
