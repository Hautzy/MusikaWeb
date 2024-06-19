import json
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

from constant import sr, hop
from utils import wv2spec_hop


def load_tensor_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    tensor = tf.convert_to_tensor(data, dtype=tf.float32)  # Convert list to TensorFlow tensor
    return tensor


def main():
    file_path = 'tensorData.json'
    tensor = load_tensor_from_json(file_path)
    print("Loaded tensor:", tensor)
    print("Tensor shape:", tensor.shape)
    print("Tensor data type:", tensor.dtype)

    # wavfile.write('test.wav', sr, np.squeeze(tensor)[:120 * sr])

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(
        np.flip(
            np.array(
                tf.transpose(
                    wv2spec_hop((tensor[:, 0] + tensor[:, 1]) / 2.0, 80.0, hop * 2),
                    [1, 0],
                )
            ),
            -2,
        ),
        cmap=None,
    )
    ax.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
