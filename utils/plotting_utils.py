import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import json
import os


def plot_training_progress(history, dataset_name, save_path):
    """Plot training metrics during eSNN training"""
    plt.figure(figsize=(15, 10))

    # Plot loss curves
    plt.subplot(2, 2, 1)
    metrics = ["loss", "dist_output_loss", "class1_output_loss", "class2_output_loss"]
    for metric in metrics:
        if metric in history.history:
            plt.plot(history.history[metric], label=metric)
    plt.title(f"Training Loss Curves - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot accuracy curves
    plt.subplot(2, 2, 2)
    metrics = [
        "dist_output_accuracy",
        "class1_output_accuracy",
        "class2_output_accuracy",
    ]
    for metric in metrics:
        if metric in history.history:
            plt.plot(history.history[metric], label=metric)
    plt.title(f"Training Accuracy Curves - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Plot combined loss trend
    plt.subplot(2, 2, 3)
    plt.plot(history.history["loss"], "b-", label="Combined Loss")
    plt.title(f"Overall Training Loss - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    figure_file = os.path.join(save_path, f"training_progress_{dataset_name}.png")
    plt.tight_layout()
    plt.savefig(figure_file)
    plt.close()

    return figure_file


def plot_from_results(results_path, save_path):
    """Plot training metrics from saved results"""
    with open(results_path, "r") as f:
        results = json.load(f)

    for dataset_name, dataset_results in results.items():
        plt.figure(figsize=(15, 10))

        # Get eSNN results for each fold
        for fold, fold_results in dataset_results.items():
            if "eSNN:adam:200:split:0.15" in fold_results:
                esnn_results = fold_results["eSNN:adam:200:split:0.15"]

                # Plot training loss
                plt.subplot(2, 2, 1)
                plt.plot(esnn_results["training_losses"], label=f"Fold {fold}")

        plt.subplot(2, 2, 1)
        plt.title(f"eSNN Training Loss - {dataset_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Plot loss distribution
        plt.subplot(2, 2, 2)
        all_final_losses = []
        all_ret_losses = []
        for fold, fold_results in dataset_results.items():
            if "eSNN:adam:200:split:0.15" in fold_results:
                esnn_results = fold_results["eSNN:adam:200:split:0.15"]
                all_final_losses.append(esnn_results["training_loss"])
                all_ret_losses.append(float(esnn_results["ret_loss"]))

        sns.boxplot(
            data=[all_final_losses, all_ret_losses],
            labels=["Training Loss", "Retrieval Loss"],
        )
        plt.title(f"Loss Distribution - {dataset_name}")

        # Plot convergence time
        plt.subplot(2, 2, 3)
        times = []
        for fold, fold_results in dataset_results.items():
            if "eSNN:adam:200:split:0.15" in fold_results:
                times.append(fold_results["eSNN:adam:200:split:0.15"]["timespent"])
        plt.bar(range(len(times)), times)
        plt.title(f"Training Time per Fold - {dataset_name}")
        plt.xlabel("Fold")
        plt.ylabel("Time (s)")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"results_analysis_{dataset_name}.png"))
        plt.close()
