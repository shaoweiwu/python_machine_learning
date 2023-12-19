import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train.shape  # df_train共1460筆資料，81種特徵
df_test.shape  # df_test共1459筆資料，80種特徵

df_train.info()  # 簡單了解df的資料形式，並去了解每一種特徵所代表的意義

df_train.columns

# 合併df_train、df_test
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_all.shape  # df_all共2919筆資料，81種特徵
# 去除Id欄位
df_all = df_all.drop(columns=["Id"])


# 對df之售價進行分析
# 描述性統計
df_train["SalePrice"].describe()

# 畫出售價的直方圖、qqplot
sns.histplot(df_train["SalePrice"], kde=True)
fig = plt.figure()
res = stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()
print("平均：", df_train["SalePrice"].mean())
print("標準差：", df_train["SalePrice"].std())
print("偏度：", df_train["SalePrice"].skew())  # 偏度>0，右偏
print("峰度：", df_train["SalePrice"].kurt())  # 峰度>3，高峽峰

# 先大致分為類別、數值變數
cat_features = df_all.select_dtypes(include=["object"]).columns
num_features = df_all.select_dtypes(exclude=["object"]).columns

len(cat_features)
len(num_features)

# 畫出數值型變數之相關係數圖
plt.figure(figsize=(12, 8))
sns.heatmap(df_all[num_features].corr())
# 可以看到['1stFlrSF', 'TotalBsmtSF']、['TotRmsAbvGrd', 'GrLivArea']、['GarageYrBlt', 'YearBuilt']、['GarageArea', 'GarageCars']兩兩之間的相關係數較高

# 找出跟售價相關係數較高的 k-1 個特徵
# k = 11
# df_train[num_features].corr(method="pearson")["SalePrice"].abs().nlargest(k)


# 畫出所有數值變數的直方圖
df_all[num_features].hist(figsize=(16, 20), bins=50)

df_all = df_all.drop(columns=["SalePrice"])

# ------------------------------------------------------
# 處理遺失值
# 先找出有遺失值的欄位
col_null = df_all.isnull().sum()
col_null[col_null > 0].index
# ['MSZoning', 'LotFrontage', 'Alley', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType']
df_all[
    [
        "MSZoning",
        "LotFrontage",
        "Alley",
        "Utilities",
        "Exterior1st",
        "Exterior2nd",
        "MasVnrType",
        "MasVnrArea",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinSF1",
        "BsmtFinType2",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "Electrical",
        "BsmtFullBath",
        "BsmtHalfBath",
        "KitchenQual",
        "Functional",
        "FireplaceQu",
        "GarageType",
        "GarageYrBlt",
        "GarageFinish",
        "GarageCars",
        "GarageArea",
        "GarageQual",
        "GarageCond",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "SaleType",
    ]
].isnull().sum()

df_all["MSZoning"].value_counts()
df_all["MSZoning"] = df_all["MSZoning"].fillna(df_all["MSZoning"].mode()[0])
# 因為MSZoning住宅類型大多為RL，因此以眾數填值

df_all["LotFrontage"] = df_all.groupby(["Neighborhood"])["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)
# LotFrontage房產與街區距離，相同區域的LotFrontage應該會類似，所以以相同Neighborhood的LotFrontage中位數填值

df_all["Alley"] = df_all["Alley"].fillna("None")
# 數據描述中NA表示沒有巷弄到達，因此填為None

df_all["Utilities"].value_counts()
df_train["Utilities"].value_counts()
df_test["Utilities"].value_counts()
df_all = df_all.drop(columns=["Utilities"])
# Utilities在test資料集中只有AllPub一種，因此決定直接drop掉

df_all.groupby(["Neighborhood", "Exterior1st"])["Exterior1st"].count()
df_all["Exterior1st"] = df_all.groupby(["Neighborhood"])["Exterior1st"].transform(
    lambda x: x.fillna(x.mode()[0])
)
df_all["Exterior2nd"] = df_all.groupby(["Neighborhood"])["Exterior2nd"].transform(
    lambda x: x.fillna(x.mode()[0])
)
# 推測相同Neighborhood的建材Exterior可能會類似，因此以Neighborhood的眾數填值

df_all["MasVnrType"] = df_all["MasVnrType"].fillna("None")
df_all["MasVnrArea"] = df_all["MasVnrArea"].fillna(0)
# 房產很可能沒有磚石貼面，因此將MasVnrType、MasVnrArea的遺失值填入None、0

for bsmt in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
    df_all[bsmt] = df_all[bsmt].fillna("None")
for bsmt in (
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "TotalBsmtSF",
):
    df_all[bsmt] = df_all[bsmt].fillna(0)
# 數據描述中NA表示沒有地下室，因此將地下室相關的遺失值填入None、0

df_all["Electrical"].value_counts()
df_all.groupby(["Neighborhood", "Electrical"])["Electrical"].count()
df_all["Electrical"] = df_all["Electrical"].fillna(df_all["Electrical"].mode()[0])
# 絕大部分的電力系統為Standard Circuit Breakers & Romex，因此Electrical以眾數填值

df_all["KitchenQual"].value_counts()
df_all.groupby(["Neighborhood", "KitchenQual"])["KitchenQual"].count()
df_all["KitchenQual"] = df_all.groupby(["Neighborhood"])["KitchenQual"].transform(
    lambda x: x.fillna(x.mode()[0])
)
# 推測相同Neighborhood的廚房品質KitchenQual可能會類似，因此以Neighborhood的眾數填值

df_all["Functional"].value_counts()
df_all["Functional"] = df_all["Functional"].fillna(df_all["Functional"].mode()[0])
# 絕大部分房屋功能Functional為Typ，因此Functional以眾數填值

df_all["FireplaceQu"] = df_all["FireplaceQu"].fillna("None")
# 數據描述中NA表示沒有壁爐，因此壁爐品質FireplaceQu遺失值以None填值

for garage in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
    df_all[garage] = df_all[garage].fillna("None")
for garage in ("GarageYrBlt", "GarageCars", "GarageArea"):
    df_all[garage] = df_all[garage].fillna(0)
# 數據描述中NA表示沒有車庫，因此將車庫相關的遺失值填入None、0

df_all["PoolQC"] = df_all["PoolQC"].fillna("None")
# 數據描述中NA表示沒有泳池，因此泳池品質PoolQC遺失值以None填值

df_all["Fence"] = df_all["Fence"].fillna("None")
# 數據描述中NA表示沒有圍籬，因此圍籬品質Fence遺失值以None填值

df_all["MiscFeature"] = df_all["MiscFeature"].fillna("None")
# 數據描述中NA表示沒有其他未涵蓋的雜項，因此其他未涵蓋的雜項MiscFeature遺失值以None填值

df_all["SaleType"].value_counts()
df_all["SaleType"] = df_all["SaleType"].fillna(df_all["SaleType"].mode()[0])
# 絕大部分銷售類型SaleType為WD，因此Functional以眾數填值

sum(df_all.isnull().any())  # 以遺失值欄位總數再次確認是否還有遺失值

# ---------------------------------------------------

# 區分可排序的類別變數
# 利用labelencoder、get_dummies處理類別資料

# 將一些屬於類別資料的數值變數型態改為str
df_all["MSSubClass"] = df_all["MSSubClass"].astype(str)
df_all["YearBuilt"] = df_all["YearBuilt"].astype(str)
df_all["YearRemodAdd"] = df_all["YearRemodAdd"].astype(str)
df_all["MoSold"] = df_all["MoSold"].astype(str)
df_all["YrSold"] = df_all["YrSold"].astype(str)

# 對類別資料中可能的有序變數執行LabelEncoding
from sklearn.preprocessing import LabelEncoder

cols = [
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "LandSlope",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "HeatingQC",
    "CentralAir",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
]
for col in cols:
    le = LabelEncoder()
    le.fit(df_all[col])
    df_all[col] = pd.DataFrame(le.transform(df_all[col]))


num_features = df_all.select_dtypes(exclude="object").columns
cat_features = df_all.select_dtypes(include="object").columns
# skewness = pd.DataFrame(df_all[num_features].apply(lambda x: x.skew()), columns=['Skewness'])
# skewness.sort_values(by='Skewness', ascending=False)

# 直接對所有數值變數取log(1+p)使其更趨近於常態分配
df_all[num_features] = np.log1p(df_all[num_features])

# 對類別變數使用get_dummies
df_all = pd.get_dummies(df_all, columns=cat_features)

# 將df_all拆成X_train、X_test
X_train = df_all[: len(df_train)].values
X_test = df_all[len(df_train) :].values

# 將df_train中的售價取對數，讓售價更趨近於常態分配
df_train["SalePrice"] = np.log(df_train["SalePrice"])
sns.histplot(df_train["SalePrice"], kde=True)
fig = plt.figure()
res = stats.probplot(df_train["SalePrice"], plot=plt)
plt.show()
print("平均：", df_train["SalePrice"].mean())
print("標準差：", df_train["SalePrice"].std())
print("偏度：", df_train["SalePrice"].skew())
print("峰度：", df_train["SalePrice"].kurt())

y_train = df_train["SalePrice"]


# -----------------------------------------
# modeling
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet


def r2_rmse(model):  # 採用K折交叉驗證，並以R2 score判定係數、RMSE作為模型衡量指標
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}
    cv_scores = cross_validate(
        model, X_train, y_train, scoring=scoring, cv=kf, return_train_score=True
    )
    print(f"迴歸模型：{model}, 折數: {n_folds}")
    print("R2_score (train): ", cv_scores["train_r2"].mean())
    print("R2_score (test): ", cv_scores["test_r2"].mean())
    print("RMSE: (train)", -cv_scores["train_neg_rmse"].mean())
    print("RMSE: (test)", -cv_scores["test_neg_rmse"].mean())


# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline

# poly = Pipeline(
#     [
#         ("poly", PolynomialFeatures(degree=2)),
#         ("linear", LinearRegression(fit_intercept=True)),
#     ]
# )
# r2_rmse(poly)

for n_folds in [2, 3, 4, 5]:
    lr = LinearRegression()
    r2_rmse(lr)  # n_folds=2 效果最佳，其餘可能overfitting
    print()
print("-" * 50)

for n_folds in [2, 3, 4, 5]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        lasso = Lasso(alpha=alpha)
        r2_rmse(lasso)
        print()
    print("-" * 50)
# 折數為2、3、4、5時，alpha=0.001對驗證集的結果皆為最佳
# 而在折數為 4 時，驗證集有較佳的 R2_score、較低的 RMSE

for n_folds in [2, 3, 4, 5]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        ridge = Ridge(alpha=alpha)
        r2_rmse(ridge)
        print()
    print("-" * 50)
# 折數為2、3、4、5時，alpha=10對驗證集的結果皆為最佳
# 而在折數為 4 時，驗證集有較佳的 R2_score、較低的 RMSE

for n_folds in [2, 3, 4, 5]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        enet = ElasticNet(alpha=alpha)
        r2_rmse(enet)
        print()
    print("-" * 50)
# 折數為2、3、4、5時，alpha=0.001對驗證集的結果皆為最佳
# 而在折數為 4 時，驗證集有較佳的 R2_score、較低的 RMSE


n_folds = 2
lr = LinearRegression()
r2_rmse(lr)

n_folds = 4
lasso = Lasso(alpha=0.001)
r2_rmse(lasso)
ridge = Ridge(alpha=10)
r2_rmse(ridge)
enet = ElasticNet(alpha=0.001)
r2_rmse(enet)

# 根據R2_score、RMSE的結果，最終決定以ElasticNet來訓練最終的模型
from sklearn.model_selection import GridSearchCV

elastic_net = ElasticNet()
param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "l1_ratio": np.linspace(0, 1, 11, endpoint=True),
}

# 使用 GridSearchCV 尋找最佳超參數組合
grid_search = GridSearchCV(
    elastic_net, param_grid, scoring="neg_mean_squared_error", cv=4
)
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

# 使用最佳模型進行預測
best_elastic_net = grid_search.best_estimator_
y_pred = best_elastic_net.predict(X_test)

df_sub = pd.DataFrame()
df_sub["Id"] = df_test.Id
df_sub["SalePrice"] = np.expm1(y_pred)  # 因為預測出來會是取完log1p的結果，因此將結果用expm1轉回來
df_sub.to_csv("elastic_net_submission.csv", index=False)  # 0.12663


# 在房價的資料集中，我們先了解資料集變數所代表的意義、資料型態，並將數值變數集類別變數加以區分開來。
# 畫出所有數值變數的相關係數圖、直方圖。
# 逐一將遺失值填入有缺失值的變數中。
# 再一次仔細地區分數值變數、類別變數，以LabelEncoder處理可排序的類別變數，以get_dummies處理不可排序的類別變數。對數值變數直接取log(1+p)，使其更趨近於常態分配以利模型訓練。
# 模型訓練中，以LinearRegression, Lasso, Ridge, ElasticNet等方法搭配交叉驗證法來訓練模型，並以R2 score、RMSE作為模型好壞的判斷依據
# 以ElasticNet來訓練最終的模型，搭配GridSearchCV來找出最佳的超參數
# 最後預測出SalePrice


# 備註
# Lasso Regression，使用L1正則化，一些參數的值可能會被壓縮到0，因此coef向量中相應的元素也會為0，代表這些特徵在模型中被認為是不重要的。可以進行特徵選擇，提高模型的性能和穩定性。
# Ridge Regression，使用L2正則化來限制模型參數的大小，從而降低模型的複雜度並避免過度擬合。L2正則化不會將參數壓縮到0，因此coef向量中的每個元素都對模型的預測有貢獻。
