# Import relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.evaluate_model import evaluate_model, plot_roc_curve

# Read data
df_train = pd.read_csv("train.csv")
print(df_train.head())
df_tested = pd.read_csv("test.csv")
print(df_tested.head())

df_submission = df_tested[["PassengerId"]]

# Evaluate data structure
df_train.info()
#  0   PassengerId  乘客ID編號
#  1   Survived     是否倖存  (0 = Dead 1 = Alive)
#  2   Pclass       船票等級  (1 = First class 2 = Second class 3 = Third class)
#  3   Name         乘客姓名
#  4   Sex          乘客性別
#  5   Age          乘客年齡
#  6   SibSp        在船上同為兄弟姊妹或配偶的數量
#  7   Parch        在船上同為家族父母或子女的數量
#  8   Ticket       船票編號
#  9   Fare         船票價格
#  10  Cabin        船艙號碼
#  11  Embarked     登船口岸  (C = Cherbourg, Q = Queenstown, S = Southampton)

# 共891筆資料，而Age、Cabin、Embarked有缺失值

df_train.describe()
# 38%的人倖存，而大多數人乘坐三等艙，年齡範圍為0-80歲，平均年齡為29.7歲

# Survived
df_train["Survived"].mean()  # 總生存率0.38

plt.figure(figsize=(6, 6))
sns.barplot(
    x=["Survived", "Died"],
    y=[df_train["Survived"].mean(), 1 - df_train["Survived"].mean()],
    hue=["Survived", "Dead"],
    palette="Set1",
)
plt.text(
    0,
    df_train["Survived"].mean() / 2,
    f"{df_train['Survived'].mean()*100: .2f}%",
    ha="right",
)
plt.text(
    1,
    (1 - df_train["Survived"].mean()) / 2,
    f"{(1-df_train['Survived'].mean())*100: .2f}%",
    ha="left",
)
plt.xlabel("Survival Status")
plt.ylabel("Survival Rate")
plt.show()

# Pclass
sum(df_train["Pclass"] == 3)  # 491位乘客搭乘三等艙
sum(df_train["Pclass"] == 2)  # 184位乘客搭乘二等艙
sum(df_train["Pclass"] == 1)  # 216位乘客搭乘一等艙
plt.figure(figsize=(6, 6))
sns.countplot(data=df_train, x="Pclass", hue="Pclass", legend=False, palette="Set1")
plt.show()
# 大部分乘客搭乘三等艙

# Survived, Pclass
df_train[df_train["Pclass"] == 1]["Survived"].mean()  # 頭等艙生存率 0.63
df_train[df_train["Pclass"] == 2]["Survived"].mean()  # 二等艙生存率 0.47
df_train[df_train["Pclass"] == 3]["Survived"].mean()  # 三等艙生存率 0.24
plt.figure(figsize=(6, 6))
sns.countplot(data=df_train[["Pclass", "Survived"]], x="Pclass", hue="Survived")
plt.legend(("Died", "Survived"))
plt.show()
# 一等艙的旅客過半數倖存，二等艙則有約一半倖存，而三等艙的旅客大多罹難

# Sex
sum(df_train["Sex"] == "male")  # 577位男性
sum(df_train["Sex"] == "female")  # 314位女性
plt.figure(figsize=(6, 6))
sns.countplot(data=df_train, x="Sex", hue="Sex", legend=False, palette="Set1")
plt.show()
sum(df_train["Sex"] == "male") / len(df_train)
# 乘客中約65%為男性，35%為女性

# Survived, Sex
df_train[df_train["Sex"] == "male"]["Survived"].mean()  # 男性生存率 0.19
df_train[df_train["Sex"] == "female"]["Survived"].mean()  # 女性生存率 0.74
plt.figure(figsize=(6, 6))
sns.countplot(data=df_train[["Sex", "Survived"]], x="Sex", hue="Survived")
plt.legend(("Died", "Survived"))
plt.show()
# 絕大部分的男性罹難，而女性大部分倖存

# Embarked
sum(df_train["Embarked"] == "C")  # 168人從Cherbourg上船
sum(df_train["Embarked"] == "Q")  # 77人從Cherbourg上船
sum(df_train["Embarked"] == "S")  # 644人從Cherbourg上船

# Survived, Embarked
df_train[df_train["Embarked"] == "C"]["Survived"].mean()  # 在Cherbourg登船生存率 0.55
df_train[df_train["Embarked"] == "Q"]["Survived"].mean()  # 在Queenstown登船生存率 0.39
df_train[df_train["Embarked"] == "S"]["Survived"].mean()  # 在Southampton登船生存率 0.34
plt.figure(figsize=(6, 6))
sns.countplot(data=df_train[["Embarked", "Survived"]], x="Embarked", hue="Survived")
plt.legend(("Died", "Survived"))
plt.show()
# 多數乘客從Southampton上船，從Cherbourg上船的乘客有較高的生存率

# Emabarked, Pclass
embarked_pclass = (
    df_train.groupby(["Embarked", "Pclass"])["Survived"].count().reset_index()
)
plt.figure(figsize=(6, 6))
sns.barplot(
    data=embarked_pclass, x="Embarked", y="Survived", hue="Pclass", palette="Set1"
)
plt.show()
# 從Cherbourg登船的旅客多為頭等艙旅客，而Queenstown則多為三等艙旅客。

# Embarked, Sex
embarked_sex = df_train.groupby(["Embarked", "Sex"])["Survived"].count().reset_index()
plt.figure(figsize=(6, 6))
sns.barplot(data=embarked_sex, x="Embarked", y="Survived", hue="Sex", palette="Set1")
plt.show()
# 就登船口岸的男女比而言，Cherbourg、Queenstown都大約1:1，Southampton則是男性較多


# plt.figure(figsize=(6, 6))
# sns.catplot(data=test, x='Sex', hue='Survived', col='Pclass', kind='count', height=6, aspect=1, palette='Set1')
# plt.show()

# Survived, Sex, Pclass
plt.figure(figsize=(6, 6))
sns.catplot(
    data=df_train,
    x="Pclass",
    hue="Survived",
    col="Sex",
    kind="count",
    height=6,
    aspect=1,
    palette="Set1",
)
plt.show()


survival_rate = df_train.groupby(["Sex", "Pclass"])["Survived"].mean().reset_index()
plt.figure(figsize=(6, 6))
ax = sns.barplot(
    data=survival_rate, x="Sex", y="Survived", hue="Pclass", palette="Set1"
)
# 設定 y 軸的上限為 1，表示生存率的最大值為 1（即 100%）
plt.ylim(0, 1)
# 排除sns在X軸上顯示0.00
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, -10),
            textcoords="offset points",
        )
plt.ylabel("Survival Rate")
plt.show()
# 在更仔細區分性別、艙等、是否倖存
# 發現只要是一、二等艙的女性絕大部分是倖存的，而三等艙的存活率也有一半
# 男性不論是哪個艙等，存活率都未過半，但一等艙的存活率是二、三等的兩倍以上

# Age, Survived
plt.figure(figsize=(6, 6))
sns.histplot(
    data=df_train[["Age", "Survived"]], x="Age", hue="Survived", bins=40, kde=True
)
plt.legend(("Survived", "Died"))
plt.show()
# 發現孩童有較高的存活數量

# Fare, Pclass
fare_pclass = df_train.groupby(["Pclass"])["Fare"].mean().reset_index()
plt.figure(figsize=(6, 6))
sns.histplot(data=df_train, x="Fare", hue="Survived", kde=True)
plt.vlines(fare_pclass["Fare"][0], 0, 100, colors="r", linestyles="dashed")
plt.text(
    fare_pclass["Fare"][0],
    100,
    "Mean Fare of Pclass 1",
    rotation=90,
    verticalalignment="bottom",
    color="r",
)
plt.vlines(fare_pclass["Fare"][1], 0, 100, colors="lime", linestyles="dashed")
plt.text(
    fare_pclass["Fare"][1],
    100,
    "Mean Fare of Pclass 2",
    rotation=90,
    verticalalignment="bottom",
    color="lime",
)
plt.vlines(fare_pclass["Fare"][2], 0, 200, colors="violet", linestyles="dashed")
plt.text(
    fare_pclass["Fare"][2],
    200,
    "Mean Fare of Pclass 3",
    rotation=90,
    verticalalignment="bottom",
    color="violet",
)
plt.show()
# 二等艙和三等艙的平均票價是差不多的，而價格越高存活的機會看起來也越高

# Fare, Age
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df_train, x="Fare", y="Age", hue="Survived", s=5)
plt.show()
# 在左下角可以發現儘管船票價格低，但許多孩童是倖存的，其餘不論年齡高低，船票價格越高存活的數量比起死亡的數量也就越多

# Fare, Age, Pclass
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df_train[df_train["Pclass"] == 1], x="Fare", y="Age", hue="Survived", s=5
)
plt.show()
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df_train[df_train["Pclass"] == 2], x="Fare", y="Age", hue="Survived", s=5
)
plt.show()
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df_train[df_train["Pclass"] == 3], x="Fare", y="Age", hue="Survived", s=5
)
plt.show()
# 發現頭等艙的孩童極少，乘客也大部分存活
# 在二等艙中全部10歲以下的孩童皆存活，然而船票價格跟是否存活沒有明顯的趨勢
# 三等艙中也發現部分孩童存活，而絕大部分的乘客死亡


# Data preprocessing
df_train.isnull().sum()
df_tested.isnull().sum()
# trian_data中Age、Cabin、Embarked有缺失值
# df_tested中Age、Fare、Cabin有缺失值
df_train["Age"] = df_train["Age"].fillna(value=df_train["Age"].mean())
df_tested["Age"] = df_tested["Age"].fillna(value=df_train["Age"].mean())
# 將Age欄位缺失值以平均值填補
df_tested["Fare"] = df_tested["Fare"].fillna(value=df_tested["Fare"].mean())
# 將Fare欄位缺失值以平均值填補
df_train["Embarked"] = df_train["Embarked"].bfill()
# 將Embarked欄位缺失值，以backfill填補
df_train = df_train.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
df_tested = df_tested.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
# 將不需要的欄位去除
df_train.isnull().sum()
df_tested.isnull().sum()
# 再次檢查資料是否含有缺失值


# df_train.info()
# df_train['Pclass'] = df_train['Pclass'].astype(object)
# df_train = pd.get_dummies(df_train)
# df_train = df_train.drop(columns=['Pclass_3', 'Sex_male', 'Embarked_S'])
# df_train['Sex'].replace(to_replace='male', value=1, inplace=True)
# df_train['Sex'].replace(to_replace='female',  value=0, inplace=True)
# df_tested['Sex'].replace(to_replace='male', value=1, inplace=True)
# df_tested['Sex'].replace(to_replace='female',  value=0, inplace=True)
# 將Sex為male的轉為1，female轉為0

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import RocCurveDisplay

X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]

features = df_train.columns.values

# 將類別型資料進行get_dummies，獲得one-hot encoding的形式
X["Pclass"] = X["Pclass"].astype(object)
X = pd.get_dummies(X)
X = X.drop(columns=["Pclass_3", "Sex_male", "Embarked_S"])

# 將數值型資料進行標準化
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
scaler = StandardScaler()
scaler.fit(X[numeric_features])
X[numeric_features] = pd.DataFrame(scaler.transform(X[numeric_features]))

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 定義 ColumnTransformer，對連續變數使用 StandardScaler，對類別變數使用 OneHotEncoder
# numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
# categorical_features = ['Pclass', 'Sex', 'Embarked']
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), numeric_features),
#         ('cat', OneHotEncoder(), categorical_features)])
# model = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', LogisticRegression())
# ])

model_LR = LogisticRegression()

# 訓練模型
model_LR.fit(X_train, y_train)

# 預測測試集
y_test_pred_LR = model_LR.predict(X_test)

# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_LR, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, X_test=X_test, y_test=y_test)

weights = pd.Series(model_LR.coef_[0], index=X.columns.values)
print(weights.sort_values(ascending=False)[:15].plot(kind="bar"))


from sklearn.svm import SVC

model_SVM = SVC(kernel="linear")
model_SVM.fit(X_train, y_train)
y_test_pred_SVM = model_SVM.predict(X_test)
# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_SVM, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, model_SVM, X_test=X_test, y_test=y_test)


from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV

preprocessor = make_pipeline(SelectKBest(f_classif, k=3))  # 僅保留好的K個變數
SVM = make_pipeline(preprocessor, SVC(random_state=42))

SVM.get_params().keys()

hyper_params_SVM = {
    "svc__gamma": [0.001, 0.0001, 0.0005],
    "svc__C": [1, 10, 100, 1000, 3000],
}

grid_SVM = GridSearchCV(
    SVM, hyper_params_SVM, scoring="recall", cv=3
)  # GridSearchCV:網格搜索, cv:交叉驗證
grid_SVM.fit(X_train, y_train)

print(grid_SVM.best_params_)

y_test_pred_grid_SVM = grid_SVM.predict(X_test)

# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_grid_SVM, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, grid_SVM, X_test=X_test, y_test=y_test)


from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier(criterion="gini")  # CART, 默認為gini
clf = model_DT.fit(X_train, y_train)

# Make predictions
y_test_pred_DT = model_DT.predict(X_test)

# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_DT, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, model_DT, X_test=X_test, y_test=y_test)

# Pruning
preprocessor = make_pipeline(SelectKBest(f_classif, k=3))  # 挑選出K個分數最高的特徵
DecisionTree = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=42))

DecisionTree.get_params().keys()

hyper_params_DT = {
    "decisiontreeclassifier__max_leaf_nodes": [10, 20, 30],  # 最多有多少個leaf nodes
    "decisiontreeclassifier__min_samples_leaf": [5, 10, 15],  # 要成為leaf nodes，最少需要多少資料
    "decisiontreeclassifier__max_depth": [5, 10, 15],  # 限制樹的高度最多幾層
}

grid_DT = GridSearchCV(DecisionTree, hyper_params_DT, scoring="recall", cv=3)
grid_DT.fit(X_train, y_train)
print(grid_DT.best_params_)

y_test_pred_grid_DT = grid_DT.predict(X_test)

# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_grid_DT, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, grid_DT, X_test=X_test, y_test=y_test)
# 在決策樹的結果中，可以發現在剪枝過後，預測能力有所提升。

# 樹的可視化
from sklearn import tree

print(tree.export_text(model_DT))


from sklearn.ensemble import RandomForestClassifier

model_RF = RandomForestClassifier()
model_RF.fit(X_train, y_train)

# Make predictions
y_test_pred_RF = model_RF.predict(X_test)

# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_RF, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, model_RF, X_test=X_test, y_test=y_test)

importances = model_RF.feature_importances_
weights = pd.Series(importances, index=X.columns.values)
weights.sort_values()[-10:].plot(kind="barh")


# Optimization
RandomForest = make_pipeline(
    SelectKBest(f_classif), RandomForestClassifier(random_state=42)
)
RandomForest.get_params().keys()

# 定義要搜索的超參數範圍
hyper_params_rf = {
    "randomforestclassifier__n_estimators": [10, 50, 100, 150, 200],
    "randomforestclassifier__max_depth": [5, 10, 15, 20],
    "randomforestclassifier__max_leaf_nodes": [10, 20, 30],
    "selectkbest__k": [3, 5, 7, 9],
}

# 使用 GridSearchCV 進行超參數搜索
grid_RF = GridSearchCV(RandomForest, hyper_params_rf, scoring="recall", cv=3)
grid_RF.fit(X_train, y_train)
print(grid_RF.best_params_)

y_test_pred_grid_RF = grid_RF.predict(X_test)
# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_grid_RF, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, grid_RF, X_test=X_test, y_test=y_test)


from xgboost import XGBClassifier

model_XG = XGBClassifier()
model_XG.fit(X_train, y_train)
y_test_pred_XG = model_XG.predict(X_test)

# 顯示混淆矩陣、模型評估結果
evaluate_model(y_test, y_test_pred_XG, class_names=[0, 1])

# 畫出ROC曲線
plot_roc_curve(model_LR, model_XG, X_test=X_test, y_test=y_test)


# 將最後所選擇的模型套用至預測上，並提交
df_tested["Pclass"] = df_tested["Pclass"].astype(object)
df_tested = pd.get_dummies(df_tested)
df_tested = df_tested.drop(columns=["Pclass_3", "Sex_male", "Embarked_S"])
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
scaler = StandardScaler()
scaler.fit(df_tested[numeric_features])
df_tested[numeric_features] = pd.DataFrame(
    scaler.transform(df_tested[numeric_features])
)
predictions = model_LR.predict(df_tested)
df_submission = pd.DataFrame(
    {"PassengerId": df_submission.PassengerId, "Survived": predictions}
)
df_submission.to_csv("submission.csv", index=False)  # 0.77511

# 在鐵達尼號的資料集中，我們藉由資料分析來了解不同因素(例如，艙等、性別)對於鐵達尼號乘客存活率的影響。
# 我們發現頭、二、三等艙的存活率分別為63%、47%、24%，艙等級別越高生存率也越高。
# 另外發現女性存活率高於男性，而孩童的生存率也明顯較高。
# 可以說造成生存率的可能因素與當時優先疏散女性、孩童的做法有關。
# 鐵達尼艙等是由上而下分配，頭等艙就位於鐵達尼上層，這可以說明為什麼艙等、船票價格越高，存活率越高。

# 在所有的訓練結果中，羅吉斯回歸與網格搜索的隨機森林準確率比其他模型都還要高，就ROC曲線之觀察、運算時間而言，羅吉斯回歸的表現較好，因此選擇羅吉斯回歸作為最終的模型選擇。
# 就羅吉斯回歸的訓練結果，對於測試集的預測準確度約為81%。性別、艙等對於預測的權重明顯較高。年齡的權重為負值，可以解釋年齡越低其生存率就越高。
# 在網格搜索的隨機森林預測結果中，可以發現船票價格、性別及年齡等數值變數的權重較高。

# 由於在年齡、船票價格、甲板、登船口岸的資料上有些缺失值，我們使用填補法填補，但可能會使資料跟真實資料間產生差異，未來可以更仔細的去挑選更適合的填補法。
# 針對資料分析的部分，則可以更深入地去探討姓名稱謂、有親人在船上的數量多寡等跟生存率之間的關係。
# 模型的部分則可以進一步調整參數進行最佳化，以獲得更好的預測結果。
