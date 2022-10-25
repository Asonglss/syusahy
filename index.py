
if __name__ == '__main__':
    iris = sklearn.datasets.load_iris()
    # data对应了样本的4个特征，150行4列
    print('>> shape of data:')
    print(iris.data.shape)
    # 显示样本特征的前5行
    print('>> line top 5:')
    print(iris.data[:5])
    # target对应了样本的类别（目标属性），150行1列
    print('>> shape of target:')
    print(iris.target.shape)
    # 显示所有样本的目标属性
    print('>> show target of data:')
    print(iris.target)