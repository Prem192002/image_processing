import tensorflow as tf

def marine_life_image_processing(image):
  # Convert the image to grayscale.
  grayscale_image = tf.image.rgb_to_grayscale(image)

  # Apply a Gaussian blur to the image.
  blurred_image = tf.image.gaussian_blur(grayscale_image, [5, 5])

  # Apply a threshold to the image.
  thresholded_image = tf.image.threshold(blurred_image, 128, 255, tf.image.NEAREST_NEIGHBOR)

  # Convert the image back to RGB.
  rgb_image = tf.image.rgb_to_grayscale(thresholded_image)

  return rgb_image

def main():
  # Load the image.
  image = tf.io.read_file("C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1")

  # Process the image.
  processed_image = marine_life_image_processing(image)

  # Save the processed image.
  tf.io.write_file("C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\processed_image.jpeg", processed_image)

if __name__ == "__main__":
  main()