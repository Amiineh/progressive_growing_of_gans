
import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
#from tensorflow.python.compiler.tensorrt import trt


def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2, axis=0)
    v2_normalized = v2 * (v1_norm / v2_norm)
    
    vectors = []
    for step in range(num_steps):
        interpolated = v1 + np.ones([512]) * (v2_normalized - v1) * step / (num_steps -1)
        interpolated_norm = np.linalg.norm(interpolated, axis=0)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        print (interpolated_normalized[0])
        vectors.append(interpolated_normalized)
    #print(v1_norm, tf.norm(v1).shape)
    return np.array(vectors)


def interpolate_hyperline(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(np.zeros([512]), np.ones([512]), num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    print(len(vectors), len(vectors[0]))
    return np.asarray(vectors)


def interpolate_between_vectors():
    v = np.random.randn(2, 512)
    vectors = interpolate_hypersphere(v[0], v[1], 45)
    return vectors


# Initialize TensorFlow session.
# tf.InteractiveSession()
tf.compat.v1.InteractiveSession()
result_path = "results/001-pgan-carpet-4l-1024-preset-v2-1gpu-fp32/"

# Import official CelebA-HQ networks.
#with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
with open(result_path + 'network-final.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

Gs.print_layers()
# Generate latent vectors.
# latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
# latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
latents = interpolate_between_vectors()

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save(result_path + 'img%d.png' % idx)

