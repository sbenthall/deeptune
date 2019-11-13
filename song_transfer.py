import fragment
import itertools
from transfer import *
import compose

## Name of songs to transform.
content_song = "organ1"
style_song = "organ2"

generated_song = "organ1over2"


# Model, and layers uses for the transfer.
# Build the extractor
song_model = tf.keras.models.load_model('song_model.h5')
content_layers = ['dense'] 
style_layers = ['conv2d_1','conv2d_2']

extractor = StyleContentModel(song_model,
                              style_layers,
                              content_layers)

### Setting up the fragment loops.

limit = None

content_frags = list(fragment.from_directory(
    song=content_song))
if limit:
    content_frags = content_frags[:limit]
## Walk the style file in a cycle next to the content
## fragments for now
style_frags = itertools.cycle(fragment.from_directory(song=style_song))

for i, (cfrag, sfrag) in enumerate(zip(content_frags,
                                       style_frags)):
    print(f"{i}/{len(content_frags)} {generated_song} Fragment Transfer")
    print(f"Content: {cfrag.song} {cfrag.number}")
    print(f"Style: {sfrag.song} {sfrag.number}")
    
    
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
                                    style_layers,
                                    style_weight=1000,
                                    content_weight=1
    )

    epochs = 7
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


compose.compose(generated_song)
