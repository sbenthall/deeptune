from transfer import *

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

## Display the content and style fragments
fig=plt.figure(figsize=(8, 8))

fig.add_subplot(1, 2, 1)
plt.imshow(content_frag.np_data)

fig.add_subplot(1, 2, 2)
plt.imshow(style_frag.np_data)

plt.show()

# Recreate the exact same model, including its weights and the optimizer
song_model = tf.keras.models.load_model('song_model.h5')
# Show the model architecture
song_model.summary()

for layer in song_model.layers:
    print(layer.name)

# Content layer where will pull our feature maps
content_layers = ['dense'] 

# Style layer of interest
style_layers = ['conv2d_1','conv2d_2']

style_extractor = model_layers(song_model,style_layers)
style_outputs = style_extractor(
    load_data.preprocess_fragment(content_frag))

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()

extractor = StyleContentModel(song_model,
                              style_layers,
                              content_layers)

results = extractor(tf.constant(
    load_data.preprocess_fragment(style_frag)
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

content_targets = extractor(load_data.preprocess_fragment(content_frag))['content']

style_targets = extractor(load_data.preprocess_fragment(style_frag))['style']

transfer_chunk = tf.Variable(load_data.preprocess_fragment(content_frag))

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs,
                                  content_targets,
                                  style_targets,
                                  num_content_layers = len(content_layers),
                                  num_style_layers = len(style_layers)
        )
        
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(image)

train_step(transfer_chunk)
train_step(transfer_chunk)
train_step(transfer_chunk)

print(transfer_chunk.numpy().shape)

transfer_frag = Fragment(
    "Mutant",
    1,
    np_data = load_data.postprocess_fragment(transfer_chunk)
)

transfer_frag.np_to_wav(save=True)

np_data = transfer(transfer_chunk,train_step)

epochs = len(np_data)

fig=plt.figure(figsize=(8, 8))

fig.add_subplot(1, epochs + 2, 1)
plt.imshow(content_frag.np_data)

for i in range(epochs):
    fig.add_subplot(1,epochs + 2, 2 + i)
    plt.imshow(np_data[i].reshape(1025,44))

fig.add_subplot(1, epochs + 2, epochs + 2)
plt.imshow(style_frag.np_data)

plt.show()
