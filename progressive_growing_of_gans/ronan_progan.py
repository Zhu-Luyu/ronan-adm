import pickle
import numpy as np
import tensorflow as tf
import PIL.Image

# Initialize TensorFlow session.
# tf.compat.v1.InteractiveSession()
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('karras2018iclr-lsun-bedroom-256x256.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate a single latent vector.
latent = np.random.RandomState(1000).randn(1, *Gs.input_shapes[0][1:]) # Generate a single random latent

# Generate dummy label (not used by the official networks).
label = np.zeros([1] + Gs.input_shapes[1][1:])

# Run the generator to produce a single image.
image = Gs.run(latent, label)

# Convert the image to PIL-compatible format.
image = np.clip(np.rint((image + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
image = image.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save the image as PNG.
PIL.Image.fromarray(image[0], 'RGB').save('img0.png')
