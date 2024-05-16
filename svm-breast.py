# 乳腺癌诊断分类

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 加载数据集
data = pd.read_csv("./data.csv")

# 数据探索
print("数据集列名：\\n", data.columns)
print("\\n数据集前5行：\\n", data.head(5))
print("\\n数据集统计描述：\\n", data.describe())

# 将特征字段分成3组
features_mean = list(data.columns[2:12])
features_se = list(data.columns[12:22])
features_worst = list(data.columns[22:32])

# 数据清洗
# 删除ID列
data.drop("id", axis=1, inplace=True)
# 将B良性替换为0，M恶性替换为1
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# 将肿瘤诊断结果可视化
sns.countplot(x='diagnosis', data=data, order=data['diagnosis'].value_counts().index)
plt.show()
# 用热力图呈现features_mean字段之间的相关性
corr = data[features_mean].corr()
plt.figure(figsize=(14, 14))
# annot=True显示每个方格的数据
sns.heatmap(corr, annot=True)
plt.show()

# 特征选择
features_remain = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean', 'symmetry_mean', 'fractal_dimension_mean']

# 抽取70%的数据作为训练集，30%作为测试集
train, test = train_test_split(data, test_size=0.3, random_state=42)
# 抽取特征选择的数值作为训练和测试数据
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']

# 采用Z-Score规范化数据，保证每个特征维度的数据均值为0，方差为1
ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)

# 创建SVM分类器
model = svm.SVC()
# 用训练集做训练
model.fit(train_X, train_y)
# 用测试集做预测
prediction = model.predict(test_X)

# 计算准确率
accuracy = metrics.accuracy_score(test_y, prediction)
print(f'预测准确率: {accuracy:.2%}')

# 计算F1分数
f1 = metrics.f1_score(test_y, prediction)
print(f'F1分数: {f1:.2f}')

