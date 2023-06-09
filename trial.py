import matplotlib as plt
import tensorflow as tf

image = tf.image.decode_image(r'input.')
equalized_image = equalize_image(image)

plt.imshow(image)
plt.show()

plt.imshow(equalized_image)
plt.show()
