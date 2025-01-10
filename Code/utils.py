#!/usr/bin/env python3

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import CFG


def res_plot(data, desc="", p=3):
    legend = ["Train", "Test"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), facecolor=(1, 1, 1))
    for i, title in enumerate(["Loss", "F1 Score"]):
        axes[i].set_ylabel(title)
    epochs = len(data["train_loss"])

    for idx, t in enumerate(["loss", "f1"]):
        x = range(1, epochs + 1)
        ax = axes[idx]
        ax.set_xticks(x[::p])
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        # title = 'Loss' if idx==0 else 'F1 Score'
        # ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.plot(x, data[f"train_{t}"], "-", label=legend[0])
        ax.plot(x, data[f"val_{t}"], "-", label=legend[1])
        c = 0
        for i, j in list(
            zip(
                [id for tup in [(i, i) for i in x[::p][:-1]] for id in tup],
                [
                    item
                    for tup in zip(
                        data[f"train_{t}"][::p][:-1],
                        data[f"val_{t}"][::p][:-1],
                    )
                    for item in tup
                ],
            ),
        ):
            c = 1 - c
            ax.annotate(
                f"{j:.3f}",
                xy=(i, j),
                rotation=60,
                va=["top", "bottom"][c] if idx else ["bottom", "top"][c],
                color=["g", "k"][c],
            )
        ax.annotate(
            f"{data[f'train_{t}'][-1]:.{max(4,p)}f}",
            xy=(epochs, data[f"train_{t}"][-1]),
            color="r",
        )
        ax.annotate(
            f"{data[f'val_{t}'][-1]:.{max(4,p)}f}",
            xy=(epochs, data[f"val_{t}"][-1]),
            color="r",
        )
        notch = (max if idx else min)(data[f"val_{t}"])
        ax.plot([notch, len(data[f"val_{t}"])], [notch, notch], "gray")
        ax.annotate(f"{notch:.4f}", xy=(1, notch), ha="right", c="black")
        ax.legend()

    metric_name = (
        CFG.images_path
        / f'plot[{datetime.now().strftime("%y%m%d%H%M")}]-Ep{epochs}B{CFG.batch_size}{desc}.png'
    )  # noqa: E501
    fig.suptitle(desc, fontsize=16)
    fig.savefig(metric_name, bbox_inches="tight", dpi=100)
    plt.show()


def flatten(tensor, sent_lens=None):
    if sent_lens is None:
        return tensor.view(-1).detach().cpu().numpy()
    # else
    return (
        torch.cat(
            [tensor[i, :length] for i, length in enumerate(sent_lens)],
        )
        .detach()
        .cpu()
        .numpy()
    )


def evaluation(y_true, y_pred, metrics):
    output = {}
    if "accuracy" in metrics:
        output["accuracy"] = accuracy_score(y_true, y_pred)
    if "f1" in metrics:
        output["f1"] = f1_score(y_true, y_pred, average="macro")
    return output


def cls_report(best_model):
    """Classification report + Confusion Matrix."""
    from data import Data, test_loader

    y_test, y_pred = [], []

    with torch.no_grad():
        for sentence, label, sent_lens in test_loader:
            scores = best_model(sentence)
            y_test.extend(flatten(label, sent_lens))
            y_pred.extend(flatten(scores.argmax(axis=-1), sent_lens))

    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=Data.labels)
    print("Accuracy is :", ac)
    print(cr)

    plt.figure(figsize=(7, 5))
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(-1, len(Data.labels))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="s",
        cmap="Blues",
        cbar=False,
        xticklabels=Data.labels,
        yticklabels=Data.labels,
        linewidths=0.1,
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    confmat_name = CFG.images_path / f'ConfusionMatrix[{datetime.now().strftime("%y%m%d%H%M")}].png'
    plt.savefig(confmat_name, dpi=100)
    plt.show()
