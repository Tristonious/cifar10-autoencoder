# Tristan Jones
# CSCI 8110 — Project 1
# Sept 2025

import argparse
import os
import time

from tensorflow.keras import callbacks

from src import (
    build_autoencoder,
    load_cifar10,
    make_datasets,
    evaluate_metrics,
    save_history_csv,
    plot_reconstructions,
    plot_training_curves,
)


def main(args):
    # Optional mixed precision for GPU runs
    if args.mixed_precision:
        try:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
            print("Mixed precision enabled (mixed_float16).")
        except Exception:
            print("Mixed precision not available; continuing with default precision.")

    # Parse filter sizes and build run tag
    filter_list = [int(x.strip()) for x in args.filters.split(",")]
    assert len(filter_list) == 2, "Provide exactly two filter sizes, e.g. --filters 32,64"
    tag = f"filters_{filter_list[0]}-{filter_list[1]}__loss_{args.loss}"

    # Build and compile model
    model = build_autoencoder(filters=tuple(filter_list))
    model.compile(optimizer="adam", loss=args.loss, metrics=[args.loss])
    model.summary()

    # Data
    x_train, x_test = load_cifar10()
    train_ds, val_ds = make_datasets(x_train, x_test, batch_size=args.batch_size, quick=args.quick)

    # Callbacks
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        callbacks.ModelCheckpoint(f"models/autoencoder_best__{tag}.h5", save_best_only=True, monitor="val_loss"),
    ]

    # Train
    start = time.time()
    history = model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=cb)
    print(f"Training finished in {time.time() - start:.1f}s")

    # Outputs
    os.makedirs("figures", exist_ok=True)
    recon_path = args.output or f"figures/recon_6__{tag}.png"
    plot_reconstructions(model, x_test, n=6, save_path=recon_path)
    plot_training_curves(history, save_path=f"figures/training_curves__{tag}.png")

    try:
        save_history_csv(history, tag, output_dir="results")
    except Exception as e:
        print("Warning: could not save history CSV:", e)

    # Evaluate
    metrics = evaluate_metrics(model, x_test, sample_n=args.eval_samples)
    print(f"[{tag}] METRICS:", metrics)
    with open(f"results/metrics__{tag}.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # Summary to stdout
    test_loss = model.evaluate(x_test, x_test, verbose=0)
    if isinstance(test_loss, (list, tuple)):
        print("Test loss:", test_loss[0])
        if len(test_loss) > 1:
            print(f"Test {args.loss}:", test_loss[1])
    else:
        print("Test loss:", test_loss)

    if metrics.get("mean_ssim") is not None:
        print(f"Mean SSIM ({args.eval_samples} samples): {metrics['mean_ssim']:.4f}")
        print(f"Mean PSNR ({args.eval_samples} samples): {metrics['mean_psnr']:.4f} dB")
    else:
        print("scikit-image not available; SSIM/PSNR not computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 convolutional autoencoder.")
    parser.add_argument("--filters", type=str, default="32,64",
                        help='Encoder filter sizes, e.g. "32,64" or "64,128"')
    parser.add_argument("--loss", type=str, default="mae", choices=["mae", "mse"],
                        help="Loss function: mae (default) or mse")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--quick", action="store_true",
                        help="Run on a small data subset for debugging")
    parser.add_argument("--mixed-precision", action="store_true", dest="mixed_precision",
                        help="Enable mixed_float16 precision (GPU)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom path for reconstruction PNG output")
    parser.add_argument("--eval-samples", type=int, default=100,
                        help="Test samples for SSIM/PSNR evaluation (0 disables)")
    args = parser.parse_args()
    main(args)
