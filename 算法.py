import numpy as np
from sklearn.decomposition import PCA
import sys


# 返回选择多少主要因素
def index_lst(lst, component=0, rate=0):
    # 组成部分：主要因素的数量
    # 比率：总和比率（主要因素）/总和（所有因素）
    # 建议费率范围：（0.8,1）
    # 如果选择rate参数，则返回index=0或小于len（lst）
    if component and rate:
        print('组件和速率只能选择一个！')
        sys.exit(0)
    if not component and not rate:
        print('组件数量参数无效！')
        sys.exit(0)
    elif component:
        print('按组件选择，组件为%s.......' % component)
        return component
    else:
        print('按速率选择，速率为%s ......' % rate)
        for i in range(1, len(lst)):
            if sum(lst[:i]) / sum(lst) >= rate:
                return i
        return 0


def main():
    # 测试数据
    mat = [[-1, -1, 0, 2, 1], [2, 0, 0, -1, -1], [2, 0, 1, 1, 0]]
    
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
    
    # 协方差矩阵
    cov_Mat = np.dot(Mat.T, Mat) / (p - 1)
    
    # 原始算法的PCA
    # 特征值递减协方差矩阵的特征值和特征向量
    U, V = np.linalg.eigh(cov_Mat)
    # 重新排列特征向量和特征值
    U = U[::-1]
    for i in range(n):
        V[i, :] = V[i, :][::-1]
    # 通过分量或速率选择特征值，而不是两者都等于0
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
    # 打印转换的数据
    print('我们选择了%d个主要因素.' % Index)
    print('PCA转换后，数据变成:\n', T1)
    
    # 基于SVD的原始算法PCA
    print('\n方法2：使用SVD的原始算法进行PCA:')
    # u： 酉矩阵，列中的特征向量
    # d： 按降序排序的奇异值列表
    u, d, v = np.linalg.svd(cov_Mat)
    Index = index_lst(d, rate=0.95)  # choose how many main factors
    T2 = np.dot(Mat, u[:, :Index])  # transformed data
    print('我们选择了%d个主要因素.' % Index)
    print('PCA转换后，数据变成:\n', T2)
    
    # Scikit学习的PCA
    pca = PCA(n_components=2)  #n分量可以是整数或浮点（0,1）
    pca.fit(mat)  # 拟合模型
    print('\n方法3：通过Scikit学习进行PCA:')
    print('PCA转换后，数据变成:')
    print(pca.fit_transform(mat))  # transformed data


main()