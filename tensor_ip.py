import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])
    return image

# Apply contrast enhancement
def enhance_contrast(image, contrast_factor):
    return tf.image.adjust_contrast(image, contrast_factor)

# Enhance water clarity by lightening
def lighten_water(image, water_mask, brightness_factor):
    # Convert the mask to a float tensor
    water_mask = tf.cast(water_mask, dtype=tf.float32)
    
    # Adjust the brightness of the water regions
    water = image + tf.expand_dims(brightness_factor * water_mask, axis=-1)
    
    # Combine the water and non-water regions
    enhanced_image = water + (1 - water_mask) * image
    return enhanced_image

# Apply color adjustment
def adjust_color(image, brightness, saturation, hue):
    image = tf.image.adjust_brightness(image, brightness)
    image = tf.image.adjust_saturation(image, saturation)
    image = tf.image.adjust_hue(image, hue)
    return image

# Apply image sharpening using a Gaussian filter
def sharpen_image(image, sigma, strength):
    kernel_size = int(2 * sigma + 1)
    kernel = create_gaussian_kernel(kernel_size, sigma, image.shape[-1])
    blurred = tf.nn.depthwise_conv2d(image[None, ...], kernel, strides=[1, 1, 1, 1], padding='SAME')
    sharpened = tf.clip_by_value(image + strength * (image - blurred[0]), 0.0, 1.0)
    return sharpened

# Create a Gaussian kernel
def create_gaussian_kernel(kernel_size, sigma, num_channels):
    kernel = np.fromfunction(
        lambda x, y, c: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - kernel_size // 2)**2 + (y - kernel_size // 2)**2) / (2 * sigma**2)),
        (kernel_size, kernel_size, num_channels)
    )
    kernel = kernel / np.sum(kernel)
    kernel = np.reshape(kernel, (kernel_size, kernel_size, num_channels, 1))
    return tf.constant(kernel, dtype=tf.float32)

# Define the parameters
image_path = r'C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg'
height = 480
width = 640
contrast_factor = 2.0
brightness_factor = 0.2  # Adjust this value to control the lightening effect on water
saturation = 1.5
hue = 0.1
sigma = 1.0
strength = 0.5

# Load and preprocess the image
image = load_image(image_path)

# Generate a mask for water regions (assuming water is represented by blue color)
water_mask = tf.less(image[..., 2], 0.5)

# Apply contrast enhancement
enhanced_image = enhance_contrast(image, contrast_factor)

# Enhance water clarity by lightening
enhanced_image = lighten_water(enhanced_image, water_mask, brightness_factor)

# Apply color adjustment
color_adjusted_image = adjust_color(enhanced_image, brightness=0.0, saturation=saturation, hue=hue)

# Apply image sharpening
sharpened_image = sharpen_image(color_adjusted_image, sigma, strength)

# Convert the images to NumPy arrays for visualization
image_np = np.array(image)
enhanced_np = np.array(enhanced_image)
color_adjusted_np = np.array(color_adjusted_image)
sharpened_np = np.array(sharpened_image)

# Display the images using matplotlib
plt.figure(figsize=(10, 8))

plt.subplot(221)
plt.title('Original')
plt.imshow(image_np)
plt.axis('off')

plt.subplot(222)
plt.title('Contrast Enhanced')
plt.imshow(enhanced_np)
plt.axis('off')

plt.subplot(223)
plt.title('Color Adjusted')
plt.imshow(color_adjusted_np)
plt.axis('off')

plt.subplot(224)
plt.title('Sharpened')
plt.imshow(sharpened_np)
plt.axis('off')

plt.tight_layout()
plt.show()
