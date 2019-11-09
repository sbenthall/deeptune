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



show_tensor(transfer_chunk)

train_step(transfer_chunk)
train_step(transfer_chunk)
train_step(transfer_chunk)

show_tensor(transfer_chunk)

print(transfer_chunk.numpy().shape)

transfer_frag = Fragment(
    "Mutant",
    1,
    np_data = load_data.postprocess_fragment(transfer_chunk)
)

transfer_frag.np_to_wav(save=True)




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
