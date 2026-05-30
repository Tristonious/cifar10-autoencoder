import tensorflow as tf
from tensorflow.keras.datasets import cifar10


def load_cifar10():
    """
    Load and normalize CIFAR-10 images to [0, 1].
    Labels are discarded — autoencoders use images only.

    Returns:
        x_train: np.ndarray, shape (50000, 32, 32, 3), float32
        x_test:  np.ndarray, shape (10000, 32, 32, 3), float32
    """
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, x_test


def make_datasets(x_train, x_test, batch_size=128, quick=False):
    """
    Build tf.data pipelines for training and validation.
    For autoencoders, inputs and targets are the same image.

    Args:
        x_train:    training images
        x_test:     validation/test images
        batch_size: int
        quick:      if True, use a small subset (1024 train / 256 val) for fast debugging

    Returns:
        train_ds, val_ds: tf.data.Dataset objects
    """
    if quick:
        x_train = x_train[:1024]
        x_test = x_test[:256]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, x_test))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds
