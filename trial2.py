import cv2
import numpy as np
import tensorflow as tf

def dehaze_dark_channel_prior(image):
    """Dehazes an image using the dark channel prior method.

    Args:
        image: The input image.

    Returns:
        The dehazed image.
    """

    # Convert the image to float32 and normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Add batch dimension to the image
    image_batch = np.expand_dims(image, axis=0)

    # Convert the image batch to tensor format
    image_tensor = tf.convert_to_tensor(image_batch)

    # Compute the dark channel of the image
    patch_size = 15
    dark_channel = tf.reduce_min(tf.image.extract_patches(images=image_tensor,
                                                          sizes=[1, patch_size, patch_size, 1],
                                                          strides=[1, 1, 1, 1],
                                                          rates=[1, 1, 1, 1],
                                                          padding='SAME'), axis=-1)

    # Estimate the atmospheric light
    num_pixels = tf.cast(tf.reduce_prod(tf.shape(dark_channel)[1:3]), tf.float32)
    num_top_pixels = tf.cast(num_pixels * 0.001, tf.int32)
    flat_dark_channel = tf.reshape(dark_channel, [-1])
    sorted_dark_channel = tf.sort(flat_dark_channel, direction='DESCENDING')
    atmospheric_light = tf.reduce_mean(sorted_dark_channel[:num_top_pixels])

    # Estimate the transmission map
    omega = 0.95
    transmission_map = 1 - omega * tf.reduce_min(tf.image.extract_patches(images=dark_channel,
                                                                         sizes=[1, patch_size, patch_size, 1],
                                                                         strides=[1, 1, 1, 1],
                                                                         rates=[1, 1, 1, 1],
                                                                         padding='SAME'), axis=-1)

    # Apply soft matting to refine the transmission map
    epsilon = 1e-6
    guided_filter_radius = 60
    guided_filter_eps = 1e-3
    transmission_map = tf.clip_by_value(transmission_map, epsilon, 1.0)
    guided_filter = tf.image.guided_filter(image_tensor[0], transmission_map, guided_filter_radius, guided_filter_eps)
    transmission_map = tf.clip_by_value(guided_filter, epsilon, 1.0)

    # Remove haze from the image
    t_min = 0.1  # Minimum transmission threshold
    dehazed_image = (image_tensor[0] - atmospheric_light) / tf.maximum(transmission_map, t_min) + atmospheric_light

    # Clip pixel values to [0, 1]
    dehazed_image = tf.clip_by_value(dehazed_image, 0.0, 1.0)

    # Convert the image back to numpy array format
    dehazed_image = dehazed_image.numpy()

    # Scale pixel values to [0, 255] and convert to uint8
    dehazed_image = (dehazed_image * 255).astype(np.uint8)

    return dehazed_image


def main():
    # Load the input image
    image = cv2.imread(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw5.jpg')

    # Apply dehazing to enhance color gradient and visibility
    dehazed_image = dehaze_dark_channel_prior(image)

    # Save the dehazed image
    cv2.imwrite(r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\img_1.jpeg', dehazed_image)

    # Display the original and dehazed images
    cv2.imshow('Original Image', image)
    cv2.imshow('Dehazed Image', dehazed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
