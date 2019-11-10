import fragment
import itertools
from transfer import *

## Name of songs to transform.
content_song = "11 - An Old Fashioned Love Song (Single Version)"
style_song = "organ1"


# Model, and layers uses for the transfer.
# Build the extractor
song_model = tf.keras.models.load_model('song_model.h5')
content_layers = ['dense'] 
style_layers = ['conv2d_1','conv2d_2']

extractor = StyleContentModel(song_model,
                              style_layers,
                              content_layers)



### building the training step-- should be moved
### to deeper code module

def train_step_factory(extractor,
                       content_targets,
                       style_targets,
                       style_content_loss_function,
                       content_layers,
                       style_layers):

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

    return tf.function(
        func = train_step
    )


### Setting up the fragment loops.

num_content_frags = 60
content_frags = list(fragment.from_directory(
    song=content_song))

## Walk the style file in a cycle next to the content
## fragments for now
style_frags = itertools.cycle(fragment.from_directory(
    song=style_song))

generated_song = "Organ Fashioned Love Song v_0_1"

for i, (cfrag, sfrag) in enumerate(zip(content_frags,
                                       style_frags)):
    print(f"{i}/{num_content_frags} {generated_song} Fragment Transfer")
    
    content_targets = extractor(
        load_data.preprocess_fragment(
            cfrag))['content']

    style_targets = extractor(
        load_data.preprocess_fragment(
            sfrag))['style']

    transfer_chunk = tf.Variable(
        load_data.preprocess_fragment(cfrag))


    train_step = train_step_factory(extractor,
                                    content_targets,
                                    style_targets,
                                    style_content_loss,
                                    content_layers,
                                    style_layers)

    epochs = 5
    np_data = transfer(transfer_chunk,
                       train_step,
                       epochs = epochs
    )
    
    data_diffs = []
    for j in range(epochs -1):
        npd = np_data[j+1] - np_data[j]
        data_diffs.append(np.abs(npd).sum().sum())

    print(f"Absolute difference per epoch: {data_diffs}")

    transfer_frag = Fragment(
        generated_song,
        i,
        np_data = load_data.postprocess_fragment(
            tf.Variable(np_data[-1]))
    )

    transfer_frag.np_to_wav(save=True)
