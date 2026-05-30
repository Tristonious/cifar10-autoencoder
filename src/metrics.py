import csv
import numpy as np
import tensorflow as tf


def evaluate_metrics(model, x_test, sample_n=100, seed=123):
    """
    Evaluate reconstruction quality on a random sample of test images.

    Computes:
      - test_loss: Keras model.evaluate() on the full test set
      - mean_ssim: mean Structural Similarity Index (requires scikit-image)
      - mean_psnr: mean Peak Signal-to-Noise Ratio in dB (requires scikit-image)

    Args:
        model:    trained Keras model
        x_test:   test images, float32 in [0, 1]
        sample_n: number of images to sample for SSIM/PSNR (0 disables)
        seed:     random seed for reproducible sampling

    Returns:
        dict with keys: test_loss, mean_ssim (None if skimage missing), mean_psnr (None if skimage missing)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        HAVE_SKIMAGE = True
    except ImportError:
        HAVE_SKIMAGE = False

    eval_out = model.evaluate(x_test, x_test, verbose=0)
    test_loss = float(eval_out[0]) if not isinstance(eval_out, dict) else float(list(eval_out.values())[0])

    results = {"test_loss": test_loss, "mean_ssim": None, "mean_psnr": None}

    if not HAVE_SKIMAGE or sample_n <= 0:
        return results

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_test), size=min(sample_n, len(x_test)), replace=False)
    originals = x_test[idx]

    recon = None
    try:
        recon = model.predict(originals, verbose=0)
    except Exception:
        try:
            recon = model.predict(np.asarray(originals, dtype=np.float32), verbose=0, batch_size=32)
        except Exception:
            ds = tf.data.Dataset.from_tensor_slices(originals).batch(32).prefetch(tf.data.AUTOTUNE)
            recon = model.predict(ds, verbose=0)

    ssim_scores, psnr_scores = [], []
    for i in range(len(originals)):
        ssim_scores.append(ssim(originals[i], recon[i], channel_axis=2, data_range=1.0))
        psnr_scores.append(psnr(originals[i], recon[i], data_range=1.0))

    results["mean_ssim"] = float(np.mean(ssim_scores))
    results["mean_psnr"] = float(np.mean(psnr_scores))
    return results


def save_history_csv(history, tag, output_dir="results"):
    """
    Save a Keras History object to <output_dir>/history__<tag>.csv for later comparison.

    Args:
        history:    Keras History object returned by model.fit()
        tag:        string identifier appended to the filename
        output_dir: directory to write the CSV (created if missing)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    hist = history.history
    keys = list(hist.keys())
    rows = list(zip(*[hist[k] for k in keys]))
    fname = os.path.join(output_dir, f"history__{tag}.csv")
    with open(fname, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for r in rows:
            writer.writerow(r)
    print(f"Saved history CSV: {fname}")
