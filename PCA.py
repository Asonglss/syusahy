import sys

import matplotlib.pyplot as plt
import np as np
import sklearn as sklearn
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

from 算法 import index_lst

iris = load_iris()
Y = iris.target
X = iris.data

pca = PCA(n_components=2)  # 实例化 n_components:降维后需要的维度，即需要保留的特征数量，可视化一般取值2
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  # 获取新矩阵


mat = X

# 测试数据的简单转换
Mat = np.array(mat, dtype='float64')
print('在PCA转换之前，数据是:\n', Mat)
print('\n方法1：原始算法PCA:')
p, n = np.shape(Mat)  # shape of Mat
t = np.mean(Mat, 0)  # 每列平均值

# 减去每列的平均值
for i in range(p):
    for j in range(n):
        Mat[i, j] = float(Mat[i, j] - t[j])
print("前5行")
print(Mat[:5])
# 协方差矩阵
cov_Mat = np.dot(Mat.T, Mat) / (p - 1)

print("协方差")
print(cov_Mat)
# 原始算法的PCA
# 特征值递减协方差矩阵的特征值和特征向量
U, V = np.linalg.eigh(cov_Mat)
# 重新排列特征向量和特征值
U = U[::-1]
for i in range(n):
    V[i, :] = V[i, :][::-1]
# 通过分量或速率选择特征值，而不是两者都等于0
print("1111111")
print(U)
print(V)
Index = index_lst(U, component=2)  # 选择多少主要因素
if Index:
    v = V[:, :Index]  # 酉矩阵子集
else:  # 不正确的费率选择可能返回Index=0
    print('费率选择无效.\n请调整费率.')
    print('费率分配如下:')
    print([sum(U[:i]) / sum(U) for i in range(1, len(U) + 1)])
    sys.exit(0)
# 数据转换
T1 = np.dot(Mat, v)
T2=T1[:5]
# 打印转换的数据
print('我们选择了%d个主要因素.' % Index)
print('PCA转换后，数据变成:\n', T2)

pca = PCA(n_components=2)  # n分量可以是整数或浮点（0,1）
pca.fit(mat)  # 拟合模型
print('\n方法3：通过Scikit学习进行PCA:')
print('PCA转换后，数据变成:')
T3=pca.fit_transform(mat)
print(T3[:5])


