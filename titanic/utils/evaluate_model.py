import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, RocCurveDisplay


def plot_confusion_matrix(y_test, y_pred, class_names=[0, 1]):  # 顯示混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

    _, ax = plt.subplots()
    sns.set(font_scale=1.2)
    sns.heatmap(df_cm, annot=True, fmt="g")

    tick_marks = np.arange(len(class_names))
    plt.tight_layout()
    plt.title("Confusion Matrix\n", y=1.1)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    ax = plt.gca()
    ax.set_xticks(tick_marks + 0.5, minor=False)
    ax.set_xticklabels(class_names, minor=False, ha="center")
    plt.tick_params(axis="x", which="both", length=0)
    ax.set_yticks(tick_marks + 0.5, minor=False)
    ax.set_yticklabels(class_names, minor=False, va="center")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xlabel("Predicted label\n")
    plt.ylabel("Actual label\n")
    plt.show()


def calculate_metrics(y_test, y_pred):  # 計算模型評估結果
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    accuracy = round(((TP + TN) / (TP + FP + TN + FN)), 3)
    precision = round((TP / (TP + FP)), 3)
    recall = round((TP / (TP + FN)), 3)
    specificity = round((TN / (TN + FP)), 3)

    return accuracy, precision, recall, specificity


def evaluate_model(y_test, y_pred, class_names=[0, 1]):
    plot_confusion_matrix(y_test, y_pred, class_names)
    accuracy, precision, recall, specificity = calculate_metrics(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall/Sensitivity: {recall}")
    print(f"Specificity: {specificity}")


def plot_roc_curve(*models, X_test, y_test):  # 畫出roc曲線
    _, ax = plt.subplots()
    plt.figure(figsize=(12, 8))
    for model in models:
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, alpha=0.8)
    plt.show()
