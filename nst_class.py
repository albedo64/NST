# import numpy, tensorflow and matplotlib
import tensorflow as tf
import numpy as np
import math
import time


# import VGG 19 model and keras Model API
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model


class NST:
  def __init__(self):
    self.model = VGG19(include_top=False, weights='imagenet')
    self.style_layer = ['block1_conv1',
                        'block3_conv1',
                        'block5_conv1'
                      ]
    self.model.trainable = False
    self.style_models = [Model(inputs=self.model.input,
                               outputs=self.model.get_layer(layer).output) for layer in self.style_layer]
    # self.model.summary()
    self.weight_of_layer = 1. / len(self.style_models)

  def gram_matrix(self, A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

  def content_cost(self, style, generated):
    J_content = 0
    a_S = self.style_models[0](style)
    a_G = self.style_models[0](generated) # Add this line to compute a_G
    GS = self.gram_matrix(a_S)
    GG = self.gram_matrix(a_G)
    content_cost = tf.reduce_mean(tf.square(GS - GG))
    J_content += content_cost * self.weight_of_layer
    return J_content

  def style_cost(self, style, generated):
    J_style = 0

    for style_model in self.style_models:
      a_S = style_model(style)
      a_G = style_model(generated)
      GS = self.gram_matrix(a_S)
      GG = self.gram_matrix(a_G)
      content_cost = tf.reduce_mean(tf.square(GS - GG))
      J_style += content_cost * self.weight_of_layer

    return J_style

  def load_and_process_image(self, image_path):
    img = load_img(image_path)
    # convert image to array
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

  def training_loop(self, content_path, style_path, iterations=50, a=10, b=1000):
    generated_images = []
    # load content and style images from their respective path
    content = self.load_and_process_image(content_path)
    style = self.load_and_process_image(style_path)
    generated = tf.Variable(content, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=7.0)

    best_cost = math.inf
    best_image = None
    for i in range(iterations):
      start_time_cpu = time.process_time()
      start_time_wall = time.time()
      with tf.GradientTape() as tape:
        J_content = self.content_cost(style, generated)
        J_style = self.style_cost(style, generated)
        J_total = a * J_content + b * J_style

      grads = tape.gradient(J_total, generated)
      opt.apply_gradients([(grads, generated)])
      end_time_cpu = time.process_time() # Record end time for CPU
      end_time_wall = time.time() # Record end time for wall time
      cpu_time = end_time_cpu - start_time_cpu # Calculate CPU time
      wall_time = end_time_wall - start_time_wall # Calculate wall time
      if J_total < best_cost:
        best_cost = J_total
        best_image = generated.numpy()
      
      print("Temps CPU: user {} µs, sys: {} ns, total: {} µs".format(
          int(cpu_time * 1e6),
          int(( end_time_cpu - start_time_cpu) * 1e9),
          int((end_time_cpu - start_time_cpu + 1e-6) * 1e6))
            )
      print("Temps: {:.2f} µs".format(wall_time * 1e6))
      print("Iteration :{}".format(i))
      print('Perte totale {:e}.'.format(J_total))
      generated_images.append(generated.numpy())

    return best_image

  def deprocess(self, img):
    # perform the inverse of the pre processing step
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # convert RGB to BGR
    img = img[:, :, ::-1]

    img = np.clip(img, 0, 255).astype('uint8')
    return img

  def display_image(self, image):
    # remove one dimension if image has 4 dimension
    if len(image.shape) == 4:
      img = np.squeeze(image, axis=0)
    else :
      img = np.squeeze(image, axis=0)

    img = self.deprocess(img)
    
    return img

