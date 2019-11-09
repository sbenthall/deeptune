from fragment import Fragment
import numpy as np
import tensorflow as tf

content_frag = Fragment(
    "11 - An Old Fashioned Love Song (Single Version)",
    100
)
content_frag.load_np_data()

style_frag = Fragment(
    "organ1",
    15
)
style_frag.load_np_data()

def prepare_fragment_np_for_model(fragment):
    np_data = fragment.np_data

    ds = np_data.shape
    np_data = np_data.reshape(1, ds[0], ds[1], 1)
    np_data = np_data.astype('float32')
    np_data/=100

    return np_data

def model_out_to_frag_np(tf_chunk):
    np_data = tf_chunk.numpy() * 100
    ds = np_data.shape
    np_data = np_data.reshape(ds[1], ds[2])

    return np_data


# Recreate the exact same model, including its weights and the optimizer
song_model = tf.keras.models.load_model('song_model.h5')
# Show the model architecture
song_model.summary()

for layer in song_model.layers:
    print(layer.name)

# Content layer where will pull our feature maps
content_layers = ['dense_1'] 

# Style layer of interest
style_layers = ['conv2d_1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def model_layers(model, layer_names):
  """ Returns a list of intermediate output values."""
  outputs = [model.get_layer(name).output for name in layer_names]

  new_model = tf.keras.Model([model.input], outputs)
  return new_model


style_extractor = model_layers(song_model,style_layers)
style_outputs = style_extractor(
    prepare_fragment_np_for_model(content_frag))

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()


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
        inputs = inputs*255.0  ## Why this?
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


extractor = StyleContentModel(song_model,
                              style_layers,
                              content_layers)

results = extractor(tf.constant(
    prepare_fragment_np_for_model(style_frag)
))

style_results = results['style']

print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())


## style transfer tutorial used up until this section:
## https://www.tensorflow.org/tutorials/generative/style_transfer#run_gradient_descent

content_targets = extractor(prepare_fragment_np_for_model(content_frag))['content']

style_targets = extractor(prepare_fragment_np_for_model(style_frag))['style']

transfer_chunk = tf.Variable(prepare_fragment_np_for_model(content_frag))

### ???
#def clip_0_1(image):
#    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight= 1   #1e-2
content_weight= 1  #1e4

def style_content_loss(outputs):
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


@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(image)

def show_tensor(tensor):
    plt.imshow(model_out_to_frag_np(tensor))
    plt.show()


show_tensor(transfer_chunk)

train_step(transfer_chunk)
train_step(transfer_chunk)
train_step(transfer_chunk)

show_tensor(transfer_chunk)

print(transfer_chunk.numpy().shape)

transfer_frag = Fragment(
    "Mutant",
    1,
    np_data = model_out_to_frag_np(transfer_chunk)
)

transfer_frag.np_to_wav(save=True)

#tensor_to_image(image)



import time
start = time.time()

epochs = 10
steps_per_epoch = 100

step = 0
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1
        train_step(transfer_chunk)
        print(".", end='')

    show_tensor(transfer_chunk)
    print("Train step: {}".format(step))
  
end = time.time()
print("Total time: {:.1f}".format(end-start))



### Next part of tutorial
### https://www.tensorflow.org/tutorials/generative/style_transfer#total_variation_loss
