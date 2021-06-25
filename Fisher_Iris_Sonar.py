import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold  # K折交叉验证库函数


def average_vector(x1, x2, n):
    m1 = np.mean(x1, axis=0).reshape(n, 1)
    m2 = np.mean(x2, axis=0).reshape(n, 1)
    # 计算不同样本的样本均值向量并将行向量转置为列向量
    return m1, m2


def in_class_dispersion_matrix(x1, x2, n, c):
    m1, m2 = average_vector(x1, x2, n)
    s1 = np.zeros((n, n))
    s2 = np.zeros((n, n))
    # 类内离散度矩阵矩阵初始化
    if c == 0:
        for loop in range(0, 49):
            s1 += (x1[loop].reshape(n, 1) - m1).dot((x1[loop].reshape(n, 1) - m1).T)
        for loop in range(0, 50):
            s2 += (x2[loop].reshape(n, 1) - m2).dot((x2[loop].reshape(n, 1) - m2).T)
    if c == 1:
        for loop in range(0, 50):
            s1 += (x1[loop].reshape(n, 1) - m1).dot((x1[loop].reshape(n, 1) - m1).T)
        for loop in range(0, 49):
            s2 += (x2[loop].reshape(n, 1) - m2).dot((x2[loop].reshape(n, 1) - m2).T)
    # 依据每类数据个数计算类内离散度矩阵s1,s2
    return s1, s2


def fisher_linear_discrimination(x1, x2, n, c):
    m1, m2 = average_vector(x1, x2, n)  # 求样本均值向量
    s1, s2 = in_class_dispersion_matrix(x1, x2, n, c)  # 求类内离散度矩阵
    s_w = s1 + s2
    # 计算总类内离散度矩阵Sw
    w = np.linalg.inv(s_w).dot(m1 - m2)
    # 计算最优投影方向 w  (公式来源及推导在《模式识别》书中p34——p36)
    m_1 = w.T.dot(m1)
    m_2 = w.T.dot(m2)
    # 在投影后的一维空间求两类的均值
    w0 = 0.5*(m_1 + m_2)
    # 计算分类阈值 w0  (为一个列向量)
    return w, w0


def classify(x, w, w0):
    y = w.T.dot(x) - w0  # 定义最终分类投影线 y
    return y


# 导入iris数据集，3类数据各50个样本，4维特征向量
iris = pandas.read_csv('iris.data', header=None, sep=',')  # 获取iris数据集, ','做间隔
iris1 = iris.iloc[0:150, 0:4]  # pandas.iloc根据行号进行索引，行号从0开始，逐次加一
iris2 = np.mat(iris1)  # 列表list转换成矩阵
Accuracy = 0
accuracy = np.zeros(10)
# 定义准确率矩阵
P1 = iris2[0:50, 0:4]      # setosa类
P2 = iris2[50:100, 0:4]    # versicolor类
P3 = iris2[100:150, 0:4]   # virginica类
G121 = np.zeros(50)
G122 = np.zeros(50)
# 初始化setosa类和versicolor类
G131 = np.zeros(50)
G132 = np.zeros(50)
# 初始化setosa类和virginica类
G231 = np.zeros(50)
G232 = np.zeros(50)
# 初始化versicolor类和virginica类

# k折交叉求证

# 第一类和第二类的线性判别
count = 0
for i in range(100):
    if i <= 49:
        data = P1[i].reshape(4, 1)
        train = np.delete(P1, i, axis=0)       # 训练样本是一个列数为t的矩阵
        W, W0 = fisher_linear_discrimination(train, P2, 4, 0)
        if (classify(data, W, W0)) >= 0:
            count += 1
            G121[i] = classify(data, W, W0)
    else:
        data = P2[i - 50].reshape(4, 1)
        train = np.delete(P2, i-50, axis=0)
        W, W0 = fisher_linear_discrimination(P1, train, 4, 1)
        if (classify(data, W, W0)) < 0:
            count += 1
            G122[i-50] = classify(data, W, W0)
Accuracy12 = count/100
print("setosa，versicolor类的分类准确率为:%.5f" % Accuracy12)

# 第一类和第三类的线性判别
count = 0
for i in range(100):
    if i <= 49:
        data = P1[i].reshape(4, 1)
        train = np.delete(P1, i, axis=0)       # 训练样本是一个列数为t的矩阵
        W, W0 = fisher_linear_discrimination(train, P3, 4, 0)
        if (classify(data, W, W0)) >= 0:
            count += 1
            G131[i] = classify(data, W, W0)
    else:
        data = P3[i - 50].reshape(4, 1)
        train = np.delete(P3, i-50, axis=0)
        W, W0 = fisher_linear_discrimination(P1, train, 4, 1)
        if (classify(data, W, W0)) < 0:
            count += 1
            G132[i-50] = classify(data, W, W0)
Accuracy13 = count/100
print("setosa，virginica类的分类准确率为:%.5f" % Accuracy13)

# 第二类和第三类的线性判别
count = 0
for i in range(100):
    if i <= 49:
        data = P2[i].reshape(4, 1)
        train = np.delete(P2, i, axis=0)       # 训练样本是一个列数为t的矩阵
        W, W0 = fisher_linear_discrimination(train, P3, 4, 0)
        if (classify(data, W, W0)) >= 0:
            count += 1
            G231[i] = classify(data, W, W0)
    else:
        data = P3[i - 50].reshape(4, 1)
        train = np.delete(P3, i-50, axis=0)
        W, W0 = fisher_linear_discrimination(P2, train, 4, 1)
        if (classify(data, W, W0)) < 0:
            count += 1
            G232[i-50] = classify(data, W, W0)
Accuracy23 = count/100
print("versicolor, virginica类的分类准确率为:%.5f" % Accuracy23)

# 画相关的图

# setosa，versicolor类
y1 = np.zeros(50)
y2 = np.zeros(50)
plt.figure(1)
plt.ylim((-0.5, 0.5))            # y坐标的范围
# 画散点图
plt.scatter(G121, y1, c='red', alpha=1, marker='.')
plt.scatter(G122, y2, c='blue', alpha=1, marker='.')
plt.xlabel('Class:1-2')
plt.show()

# setosa，virginica类
plt.figure(2)
plt.ylim((-0.5, 0.5))            # y坐标的范围
# 画散点图
plt.scatter(G131, y1, c='red', alpha=1, marker='.')
plt.scatter(G132, y2, c='blue', alpha=1, marker='.')
plt.xlabel('Class:1-3')
plt.show()

# versicolor, virginica类
plt.figure(3)
plt.ylim((-0.5, 0.5))            # y坐标的范围
# 画散点图
plt.scatter(G231, y1, c='red', alpha=1, marker='.')
plt.scatter(G232, y2, c='blue', alpha=1, marker='.')
plt.xlabel('Class:2-3')
plt.show()


# 导入sonar.all-data数据集，2类数据共208个样本，60维特征向量
sonar = pandas.read_csv('sonar.all-data', header=None, sep=',')  # 获取sonar数据集, ','做间隔
sonar1 = sonar.iloc[0:208, 0:60]  # pandas.iloc根据行号进行索引，行号从0开始，逐次加一
sonar2 = np.mat(sonar1)  # 列表list转换成矩阵matrix
Accuracy = np.zeros(60)
accuracy = np.zeros(10)
G1 = np.zeros(97)
G2 = np.zeros(111)

# 定义准确率矩阵
for N in range(1, 61):  # N是当前的维数
    for t in range(10):  # 每一维都求十次平均值
        sonar_random = (np.random.permutation(sonar2.T)).T
        P1 = sonar_random[0:97, 0:N]
        P2 = sonar_random[97:208, 0:N]
        # 对原sonar数据进行每列打乱(行转置而来)
        count = 0
        # k折交叉验证
        for i in range(208):
            if i <= 96:
                test = P1[i].reshape(N, 1)
                train = np.delete(P1, i, axis=0)  # 训练样本是一个列数为t的矩阵
                W, W0 = fisher_linear_discrimination(train, P2, N, 0)
                if (classify(test, W, W0)) >= 0:
                    count += 1
                    G1[i] = classify(test, W, W0)
            else:
                test = P2[i - 97].reshape(N, 1)
                train = np.delete(P2, i - 97, axis=0)
                W, W0 = fisher_linear_discrimination(P1, train, N, 1)
                if (classify(test, W, W0)) < 0:
                    count += 1
                    G2[i - 97] = classify(test, W, W0)
        accuracy[t] = count / 208
    for k in range(10):
        Accuracy[N - 1] += accuracy[k]
    Accuracy[N - 1] = Accuracy[N - 1] / 10
    print("当数据为%d维时，Accuracy:%.5f" % (N, Accuracy[N - 1]))
    # 绘图（分类情况）
    y1 = np.zeros(97)
    y2 = np.zeros(111)
    plt.figure(N)
    plt.ylim((-0.5, 0.5))  # y坐标的范围
    # 画散点图
    plt.scatter(G1, y1, c='yellow', alpha=1, marker='.')
    plt.scatter(G2, y2, c='blue', alpha=1, marker='.')
    plt.xlabel('Class:N')
    plt.show()

# 画accuracy相关的图
LOOP = np.arange(1, 61, 1)
plt.xlabel('dimension')
plt.ylabel('Accuracy')
plt.ylim((0.5, 0.8))  # y坐标的范围
plt.plot(LOOP, Accuracy, 'b')
plt.show()


kf = KFold(n_splits=5)
d12 = kf.split(P1 + P2)
d13 = kf.split(P1 + P3)
d23 = kf.split(P2 + P3)
for train_idx, test_idx in d12:
    train_data = np.array(d12[train_idx])
    test_data = np.array(d12[test_idx])
    print('train_index={}, train_data={}'.format(train_idx, train_data))
    print('test_index={}, test_data={}'.format(test_idx, test_data))
for i in range(100):
    if i <= 49:
        W, W0 = fisher_linear_discrimination(train_idx, P2, 4, 0)
