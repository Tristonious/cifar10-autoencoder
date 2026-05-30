import tensorflow as tf
from tensorflow.keras import layers, models


def build_autoencoder(input_shape=(32, 32, 3), filters=(32, 64)):
    """
    Convolutional autoencoder for CIFAR-10 image reconstruction.

    Encoder: two Conv2D + MaxPooling2D blocks that reduce spatial dims
             from 32x32 down to 8x8 while expanding channel depth.
    Decoder: symmetric UpSampling2D + Conv2D blocks that restore 32x32,
             with a sigmoid output layer to keep pixel values in [0, 1].

    Note: sigmoid on the final layer can cause rounding issues with
    mixed-precision (mixed_float16) training.

    Args:
        input_shape: tuple, default (32, 32, 3)
        filters: 2-tuple of ints for the two encoder conv blocks, e.g. (32, 64) or (64, 128)

    Returns:
        Compiled-ready Keras Model (call .compile() before training)
    """
    input_img = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(filters[0], (3, 3), activation="relu", padding="same")(input_img)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(filters[1], (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2D(filters[1], (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(filters[0], (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    return models.Model(input_img, decoded)
