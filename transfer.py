from fragment import Fragment
import load_data
import numpy as np
import tensorflow as tf
import time


import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False


def model_layers(model, layer_names):
  """ Returns a list of intermediate output values."""
  outputs = [model.get_layer(name).output for name in layer_names]

  new_model = tf.keras.Model([model.input], outputs)
  return new_model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, model, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.song_model = model
        self.transfer_model =  model_layers(model, style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.transfer_model.trainable = False
        
    def call(self, inputs):
        "XYZ: Expects float input in [?,?]"
        ###inputs = inputs*255.0  ## Why this?
        #preprocessed_input = self.transfer_model.preprocess_input(inputs)
        outputs = self.transfer_model(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                          outputs[self.num_style_layers:])
        
        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name:value 
                        for content_name, value 
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name:value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}
        
        return {'content':content_dict, 'style':style_dict}





### ???
#def clip_0_1(image):
#    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight= 10000   #1e-2
content_weight= 1  #1e4

def style_content_loss(outputs,
                       content_targets,
                       style_targets,
                       num_content_layers=1,
                       num_style_layers=1
):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

def show_tensor(tensor):
    plt.imshow(load_data.postprocess_fragment(tensor))
    plt.show()


def transfer(base_fragment,
             train_step_function,
             epochs = 10,
             steps_per_epoch = 100):
    
    start = time.time()

    step = 0

    np_data = []
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step_function(base_fragment)
            print(".", end='')

        np_data.append(base_fragment.numpy())
        print("Train step: {}".format(step))
  
    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    return np_data


### Next part of tutorial
### https://www.tensorflow.org/tutorials/generative/style_transfer#total_variation_loss
