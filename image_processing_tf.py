import tensorflow as tf

# Load and preprocess the image
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [256, 256])  # Resize to a desired size
    image = image / 255.0  # Normalize the image
    return image

# Enhance image visibility and clarity
def enhance_image(image):
    image = tf.image.adjust_contrast(image, 1.2)  # Increase contrast
    image = tf.image.adjust_brightness(image, 0.1)  # Increase brightness
    image = tf.image.adjust_gamma(image, gamma=1.2)  # Apply gamma correction
    image = tf.image.median_rgb(image, filter_shape=3)  # Apply median filtering
    return image

# Load the image
image_path = r"C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\input\uw1.jpeg"
image = preprocess_image(image_path)

# Enhance the image
enhanced_image = enhance_image(image)

# Convert the enhanced image to uint8 format
enhanced_image = tf.image.convert_image_dtype(enhanced_image, dtype=tf.uint8)

# Save the enhanced image
tf.io.write_file(r"C:\Users\Prem\OneDrive\Desktop\Coratia_Tech\output\enhanced_image.jpeg", tf.image.encode_jpeg(enhanced_image))

