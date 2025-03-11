import os
import argparse
import itertools

import torch
import numpy as np
import random

import app
import train
import data

def main():
    parser = argparse.ArgumentParser(description="MNIST Explorer")
    subparsers = parser.add_subparsers(dest="command")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a neural network")
    train_parser.add_argument("--epochs", type=int, nargs="+", default=[10], help="Number of epochs for training")
    train_parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.001], help="Learning rate for training")
    train_parser.add_argument("--batch-size", type=int, nargs="+", default=[64], help="Batch size for training")
    train_parser.add_argument("--val-split", type=float, nargs="+", default=[0.1], help="Validation split for training")
    train_parser.add_argument("--hidden-layer-sizes", type=int, nargs="+", default=[84], help="Hidden layer sizes for the neural network")
    train_parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training; either adam or sgd")
    train_parser.add_argument("--normalize", action="store_true", help="Normalize the data")
    train_parser.add_argument("--force-retrain", action="store_true", help="Force retrain the model for all parameters")
    train_parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")

    train_parser.add_argument("--data-dir", type=str, default="data", help="Directory to store MNIST data")
    train_parser.add_argument("--model-dir", type=str, default="models", help="Directory to store trained model data")

    # Explore subcommand
    explore_parser = subparsers.add_parser("explore", help="Explore model performance with Gradio")
    explore_parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server")
    explore_parser.add_argument("--load-models", action="store_true", help="Load models from model directory")

    explore_parser.add_argument("--data-dir", type=str, default="data", help="Directory where MNIST data is stored")
    explore_parser.add_argument("--model-dir", type=str, default="models", help="Directory where trained model data is stored")

    args = parser.parse_args()

    if args.command == "train":
        # Set random seeds
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

        # Set up directories
        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)

        settings = itertools.product(args.epochs, args.learning_rate, args.batch_size, args.val_split)
        for epochs, learning_rate, batch_size, val_split in settings:
            # Check if model already exists by creating filename path
            hl_name = "hls" + "-".join(map(str, args.hidden_layer_sizes))
            model_name = f"mnist_{hl_name}_e{epochs}_lr{learning_rate}_bs{batch_size}_vs{val_split}_{args.optimizer}_norm{args.normalize}-seed{args.random_seed}"
            model_path = os.path.join(args.model_dir, f"{model_name}.pt")

            if os.path.exists(model_path) and not args.force_retrain:
                print(f"Model {model_name} already exists. Skipping...")
            else:
                print(15*"-")
                print(f"Training with: \n Epochs: {epochs} \n Learning Rate: {learning_rate} \n Batch Size: {batch_size} \n Normalize: {args.normalize} \n Validation Split: {val_split} \n Hidden Layer Sizes: {args.hidden_layer_sizes} \n Optimizer: {args.optimizer}")
                model = train.NeuralNetwork(
                    train_config=train.TrainConfig(
                        epochs=epochs,
                        learning_rate=learning_rate,
                        optimizer=args.optimizer,
                    ),
                    data=data.MNISTData(
                        batch_size=batch_size,
                        val_split=val_split,
                        normalize=args.normalize,
                    ),
                    hidden_layer_sizes=args.hidden_layer_sizes,
                )
                loss_df, performance_df = model.run_training()

                torch.save(model.state_dict(), model_path)
                loss_df.to_csv(os.path.join(args.model_dir, f"{model_name}-loss.csv"), index=False)
                performance_df.to_csv(os.path.join(args.model_dir, f"{model_name}-performance.csv"), index=False)

    elif args.command == "explore":
        print(f"Starting Gradio server on port {args.port}")
        model_store = app.ModelStore(args.model_dir, args.load_models, args.data_dir)
        app_instance = app.App(model_store)

if __name__ == "__main__":
    main()