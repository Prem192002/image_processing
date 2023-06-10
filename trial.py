import cv2
import numpy as np
import tensorflow as tf

def dehaze_adaptive_histogram_equalization(image):
    """Dehazes an image using adaptive histogram equalization.

    Args:
        image: The input image.

    Returns:
        The dehazed image.
    """

    # Convert the image to float32 and normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Convert the image to tensor format
    image_tensor = tf.convert_to_tensor(image)

    # Estimate the transmission map using a dark channel prior
    min_filter_radius = 15
    dark_channel = tf.reduce_min(tf.image.rgb_to_grayscale(image_tensor), axis=-1)

    # Reshape the dark channel tensor to have 2 spatial dimensions
    dark_channel_reshaped = tf.reshape(dark_channel, [1, dark_channel.shape[0], dark_channel.shape[1], 1])

    transmission_map = 1 - (tf.reduce_min(tf.nn.max_pool2d(dark_channel_reshaped, min_filter_radius, strides=1, padding='SAME'), axis=[1, 2, 3]) / 255.0)

    # Estimate the atmospheric light using the highest intensity pixel
    top_percentile = 0.001
    num_pixels = tf.cast(tf.size(dark_channel), tf.float32)
    num_top_pixels = tf.cast(tf.round(num_pixels * top_percentile), tf.int32)
    sorted_dark_channel = tf.sort(tf.reshape(dark_channel, [-1]))
    atmospheric_light = tf.reduce_mean(sorted_dark_channel[-num_top_pixels:])

    # Estimate the scene radiance
    epsilon = 0.001
    scene_radiance = (image_tensor - atmospheric_light) / tf.maximum(transmission_map, epsilon) + atmospheric_light

    # Clip pixel values to [0, 1]
    scene_radiance = tf.clip_by_value(scene_radiance, 0.0, 1.0)

    # Convert the image back to numpy array format
    dehazed_image = scene_radiance.numpy()

    # Scale pixel values to [0, 255] and convert to uint8
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    return dehazed_image

def main():
    # Load the input image.
    image = cv2.imread(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg')

    # Dehaze the image.
    dehazed_image = dehaze_adaptive_histogram_equalization(image)

    # Save the dehazed image.
    output_path = r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\pre1.jpeg'
    cv2.imwrite(output_path, dehazed_image)
    print(f"Dehazed image saved at: {output_path}")

if __name__ == '__main__':
    main()

