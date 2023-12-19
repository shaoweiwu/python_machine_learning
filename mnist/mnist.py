import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train.shape  # (42000, 785)
df_test.shape  # (28000, 784)

# 檢查是否有遺失值
sum(df_train.isnull().any())  # 0，沒有遺失值
sum(df_test.isnull().any())  # 0，沒有遺失值

df_train.head()
X_train = df_train.iloc[:, 1:].values  #
y_train = df_train.iloc[:, 0]  #
X_test = df_test.values

ax = sns.countplot(x=y_train, color="#3081D0")
ax.set(title="Target Distribution")
ax.axhline(y=4200, linestyle="--", color="r")
plt.show()
# 可以看出數字0-9的數量分配還算平均


# 正規化，公式(x-x_min)/(x_max-x_min)，因為圖片的像素值為0-255，所以直接除以255.0
X_train = X_train / 255.0
X_test = X_test / 255.0

# 畫出0-9每個數字
fig, ax = plt.subplots(2, 5, figsize=(12, 4))
for num in range(0, 10):
    idx = y_train[y_train == num].index[0]
    image = X_train[idx].reshape(28, 28)
    ax[(num // 5), (num % 5)].imshow(X=image)
    ax[(num // 5), (num % 5)].set_title(f"Label: {num}")
    ax[(num // 5), (num % 5)].axis("off")
plt.show()

# --------------------------------------------------------
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

knn_clf = KNeighborsClassifier()
knn_clf_params = {"n_neighbors": [1, 3, 5, 7, 9]}
knn_clf_gs = GridSearchCV(
    estimator=knn_clf,
    param_grid=knn_clf_params,
    scoring="accuracy",
    cv=5,
    verbose=1,
    return_train_score=True,
)
knn_clf_gs.fit(X_train, y_train)
print("Best KNN params:", knn_clf_gs.best_params_)  # {'n_neighbors': 3}
print("Best KNN cv accuracy:", knn_clf_gs.best_score_)  # 0.96669

# 畫出混淆矩陣
best_est = knn_clf_gs.best_estimator_
y_pred = best_est.predict(X_train)
conf_matrix = confusion_matrix(y_train, y_pred)
conf_matrix_disp = ConfusionMatrixDisplay(conf_matrix)
conf_matrix_disp.plot(cmap="Blues")

# 畫出訓練集、驗證集的準確率分數
fig, ax = plt.subplots()
ax.plot(
    range(len(knn_clf_params["n_neighbors"])),
    knn_clf_gs.cv_results_["mean_train_score"],
    label="train",
    color="blue",
)
ax.plot(
    range(len(knn_clf_params["n_neighbors"])),
    knn_clf_gs.cv_results_["mean_test_score"],
    label="test",
    color="orange",
)
ax.set_xticks(
    ticks=range(len(knn_clf_params["n_neighbors"])),
    labels=knn_clf_params["n_neighbors"],
)

best_n = knn_clf_params["n_neighbors"].index(knn_clf_gs.best_params_["n_neighbors"])

plt.axvline(x=best_n, linestyle="--", color="red", label="best", alpha=0.3)
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.title("PCA+KNN GridSearchCV Results")
plt.show()

# 運用最佳參數訓練模型，並對Test資料集進行預測
best_params = knn_clf_gs.best_params_
knn_clf_best = KNeighborsClassifier(**best_params)
knn_clf_best.fit(X_train, y_train)
knn_pred = knn_clf_best.predict(X_test)
df_sub = pd.DataFrame({"ImageId": range(1, len(df_test) + 1), "Label": knn_pred})
df_sub.to_csv("knn_3_submission.csv", index=False)  # 0.96803

# -------------------------------------------------
# PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=784, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_train_pca.shape

df_train_pca = pd.DataFrame(data=X_train_pca, columns=np.arange(0, 784).astype(str))
df_train_pca["label"] = y_train

# 畫出前兩大主成分所構成的散點圖
fig, ax = plt.subplots()
ax = sns.scatterplot(
    data=df_train_pca, x="0", y="1", hue="label", s=10, alpha=0.5, palette="Set1"
)
ax.set(xlabel="1st_Comp", ylabel="2nd_Comp")
plt.show()


pca.fit(X_train)
np.round(pca.explained_variance_ratio_, 3)

# 畫出累積解釋變異量
var_plot = [0] + pca.explained_variance_ratio_.tolist()[:100]
cum_explained_var_ratio = np.cumsum(var_plot)
plt.plot(cum_explained_var_ratio)
plt.xlabel("# principal components")
plt.ylabel("cumulative explained variance")
plt.show()

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()
target_variance_ratio = 0.90
num_components = (
    np.argmax(cumulative_explained_variance >= target_variance_ratio) + 1
)  # 找出能解釋9成以上變異的主成分個數
sum(pca.explained_variance_ratio_.tolist()[:87])


# 畫出經過PCA處理過後的圖
pca = PCA(n_components=87, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_train_pca.shape

fig, ax = plt.subplots(nrows=10, ncols=2, figsize=(6, 20))

for i in range(10):
    idx = np.where(y_train == i)[0][0]
    original_img = ax[i, 0].imshow(X=X_train[idx].reshape(28, 28))
    original_img.axes.get_yaxis().set_visible(False)
    original_img.axes.get_xaxis().set_visible(False)
    original_img.axes.set_title(f"Original [{i}]:")

    approx_img_ = pca.inverse_transform(X=X_train_pca[idx])
    approx_img = ax[i, 1].imshow(X=approx_img_.reshape(28, 28))
    approx_img.axes.get_yaxis().set_visible(False)
    approx_img.axes.get_xaxis().set_visible(False)
    approx_img.axes.set_title(f"Approximation [{i}]:")

# --------------------------------------------------
# 運用KNN搭配PCA來訓練模型
knn_clf = Pipeline(
    steps=[("pca", PCA(n_components=num_components)), ("knn", KNeighborsClassifier())]
)

knn_clf_params = {"knn__n_neighbors": [1, 3, 5, 7, 9]}

knn_clf_gs = GridSearchCV(
    estimator=knn_clf,
    param_grid=knn_clf_params,
    scoring="accuracy",
    cv=5,
    verbose=1,
    return_train_score=True,
)

knn_clf_gs.fit(X=X_train, y=y_train)

print(
    "best PCA+KNN params:", knn_clf_gs.best_params_
)  # best PCA+KNN params: {'knn__n_neighbors': 3}
print("best PCA+KNN cv accuracy:", knn_clf_gs.best_score_)
# best PCA+KNN cv accuracy: 0.97033

best_est = knn_clf_gs.best_estimator_
y_pred = best_est.predict(X_train)
conf_matrix = confusion_matrix(y_train, y_pred)
conf_matrix_disp = ConfusionMatrixDisplay(conf_matrix)
conf_matrix_disp.plot(cmap="Blues")

# 畫出訓練集、驗證集的準確率分數
fig, ax = plt.subplots()
ax.plot(
    range(len(knn_clf_params["knn__n_neighbors"])),
    knn_clf_gs.cv_results_["mean_train_score"],
    label="train",
    color="blue",
)
ax.plot(
    range(len(knn_clf_params["knn__n_neighbors"])),
    knn_clf_gs.cv_results_["mean_test_score"],
    label="test",
    color="orange",
)
ax.set_xticks(
    ticks=range(len(knn_clf_params["knn__n_neighbors"])),
    labels=knn_clf_params["knn__n_neighbors"],
)

best_n = knn_clf_params["knn__n_neighbors"].index(
    knn_clf_gs.best_params_["knn__n_neighbors"]
)

plt.axvline(x=best_n, linestyle="--", color="red", label="best", alpha=0.3)
plt.legend()
plt.xlabel("n_neighbors")
plt.ylabel("Accuracy")
plt.title("PCA+KNN GridSearchCV Results")
plt.show()

# 運用最佳參數訓練模型，並對Test資料集進行預測
best_params = knn_clf_gs.best_params_
knn_clf_best = Pipeline(
    steps=[
        ("pca", PCA(n_components=87)),
        ("knn", KNeighborsClassifier(best_params["knn__n_neighbors"])),
    ]
)
knn_clf_best.fit(X_train, y_train)
knn_pred = knn_clf_best.predict(X_test)
df_sub = pd.DataFrame({"ImageId": range(1, len(df_test) + 1), "Label": knn_pred})
df_sub.to_csv("pca_knn_submission.csv", index=False)  # 0.97321

# --------------------------------------------
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

train_img = X_train.reshape((42000, 28, 28, 1)).astype("float32")
test_img = X_test.reshape((28000, 28, 28, 1)).astype("float32")

train_label = to_categorical(y_train)

X_train, X_val, y_train, y_val = train_test_split(
    train_img, train_label, test_size=0.2, random_state=42
)

model = models.Sequential()
model.add(
    layers.Conv2D(
        filters=16, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)
    )
)
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
model_checkpoint = ModelCheckpoint("cnn_model.keras", save_best_only=True)

history = model.fit(
    X_train,
    y_train,
    epochs=60,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, model_checkpoint],
)

# 模型表現分析
plt.plot(history.history["accuracy"], label=str("Training " + "accuracy"))
plt.plot(history.history["val_accuracy"], label=str("Validation " + "val_accuracy"))
plt.legend()
plt.show()

plt.plot(history.history["loss"], label=str("Training " + "loss"))
plt.plot(history.history["val_loss"], label=str("Validation " + "val_loss"))
plt.legend()
plt.show()

# model.save("cnn_model.h5")
# model = load_model("cnn_model.keras")
cnn_pred = model.predict(test_img)

pred = []
for i in cnn_pred:
    pred.append(np.argmax(i))

df_sub = pd.DataFrame({"ImageId": range(1, len(df_test) + 1), "Label": pred})
df_sub.to_csv("cnn_submission.csv", index=False)  # 0.99235

# 在數字辨識的資料集中，我們首先了解資料是否有缺失值，並確認數字的出現比例是否差不多
# 再來將像素資料進行正規化
# 運用GridSearchCV搭配KNN分類法訓練模型，並畫出混淆矩陣、訓練時訓練集、測試集的準確率分數
# 進一步用PCA進行降維，畫出降維後的圖形、主成分的累積解釋變異量
# 搭配PCA，再一次使用KNN訓練模型
# 使用tensorflow中的keras套件，運用CNN卷積神經網路進行訓練，並得到最終的預測結果
