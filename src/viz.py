import csv
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_reconstructions(model, x_test, n=6, save_path=None):
    """
    Plot original vs reconstructed images side by side.

    Args:
        model:     trained Keras model
        x_test:    test images, float32 in [0, 1]
        n:         number of image pairs to show (default 6)
        save_path: if provided, save PNG to this path instead of displaying
    """
    decoded_imgs = model.predict(x_test[:n])
    plt.figure(figsize=(12, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_training_curves(history, save_path=None):
    """
    Plot loss and metric curves from a Keras History object.

    Skips duplicate lines: if a metric is numerically identical to loss
    (common when loss=mae and mae is also tracked), only one line is drawn.

    Args:
        history:   Keras History object returned by model.fit()
        save_path: if provided, save PNG to this path instead of displaying
    """
    plt.figure(figsize=(8, 4))
    hist = history.history

    if "loss" in hist:
        plt.plot(hist["loss"], label="train_loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="val_loss")

    skip_keys = {"loss", "val_loss", "lr", "learning_rate"}
    base_metrics = [k for k in hist if not (k in skip_keys or k.startswith("val_"))]

    for m in base_metrics:
        try:
            if "loss" in hist and np.allclose(np.array(hist["loss"]), np.array(hist[m])):
                continue
        except Exception:
            pass
        plt.plot(hist[m], label=f"train_{m}")
        val_key = f"val_{m}"
        if val_key in hist:
            try:
                if "val_loss" in hist and np.allclose(np.array(hist["val_loss"]), np.array(hist[val_key])):
                    continue
            except Exception:
                pass
            plt.plot(hist[val_key], label=f"val_{m}")

    plt.xlabel("epoch")
    plt.ylabel("Loss / Metric")
    plt.title("Training Curves")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def load_history_csv(path):
    """
    Load a history__<tag>.csv file into a dict of {metric: [values]}.

    Args:
        path: path to CSV file

    Returns:
        dict mapping column names to lists of float values
    """
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}
    keys = rows[0]
    data = {k: [] for k in keys}
    for r in rows[1:]:
        for k, v in zip(keys, r):
            try:
                data[k].append(float(v))
            except Exception:
                data[k].append(np.nan)
    return data


def compare_training_curves(csv_paths, metric="loss", save_path=None):
    """
    Overlay training curves from multiple history CSV files on one plot.

    Args:
        csv_paths: list of paths to history__<tag>.csv files
        metric:    metric name to plot, e.g. 'loss' or 'mae'
        save_path: if provided, save PNG to this path instead of displaying
    """
    plt.figure(figsize=(8, 4))
    for p in csv_paths:
        if not os.path.exists(p):
            print("Missing file:", p)
            continue
        hist = load_history_csv(p)
        tag = os.path.splitext(os.path.basename(p))[0].replace("history__", "")
        if metric in hist:
            plt.plot(hist[metric], label=f"{tag} train_{metric}")
        if f"val_{metric}" in hist:
            plt.plot(hist[f"val_{metric}"], linestyle="--", label=f"{tag} val_{metric}")

    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"Compare {metric} across runs")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print("Saved comparison plot to", save_path)
    else:
        plt.show()
    plt.close()
