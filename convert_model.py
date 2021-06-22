import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
#from tensorflow.python.compiler.tensorrt import trt


tf.compat.v1.InteractiveSession()
result_path = "results/001-pgan-carpet-4l-1024-preset-v2-1gpu-fp32/"

# Import official CelebA-HQ networks.
#with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
with open(result_path + 'network-final.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

tf.train.write_graph(Gs, result_path,
                     'network_final.pb', as_text=False)

