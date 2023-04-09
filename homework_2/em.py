import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def data_processing_csv(file_path):
    """
    获取待处理的文件数据
    :param file_path: csv文件名对应路径
    :return data: 样本数据
    """
    data = []
    with open(file_path, 'r') as f:
        for i in f.readlines():
            if i != "height\n":
                data.append(eval(i.strip()))
    return data


def gmm_em(data, k, alpha, mu, sigma, tol, max_step):
    """
    用em算法求解高斯混合模型参数
    :param data: 样本数据
    :param k: 混合模型个数
    :param alpha: 第k个模型在混合模型中发生的概率(k维)
    :param mu: 第k个模型的期望(k维)
    :param sigma: 第k个模型的标准差(k维)
    :param tol: 参数变化小于该值时代表收敛，算法结束
    :param max_step: 最大循环求解次数
    :return [alpha, mu, sigma]: 高斯混合模型参数[每个子模型的期望,标准差,在混合模型中发生的概率]
    """
    num = len(data)
    likelihood_old = 0
    data_2d = np.zeros((num, k))
    for i in range(num):
        data_2d[i, :] = [data[i]] * k
    for i in range(max_step):
        # E步
        gamma = np.zeros((num, k))  # 每个数据来自各个子模型的概率
        for j in range(k):
            gamma[:, j] = alpha[j] * norm.pdf(data, mu[j], sigma[j])
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)  # 求解每个数据来自各个子模型的概率

        # M步
        mu = np.sum(gamma * data_2d, axis=0) / np.sum(gamma, axis=0)  # 更新期望
        sigma = np.sqrt(np.sum(gamma * (data_2d - mu) ** 2, axis=0) / np.sum(gamma, axis=0))  # 更新标准差
        alpha = np.sum(gamma, axis=0) / num  # 更新子模型在混合模型中发生的概率

        # 计算对数似然
        likelihood_new = sum(np.log(sum(alpha[j] * norm.pdf(data, mu[j], sigma[j]) for j in range(k))))

        # 检查收敛性
        if np.abs(likelihood_new - likelihood_old) < tol:
            break
        likelihood_old = likelihood_new

    return [alpha, mu, sigma]


if __name__ == "__main__":
    k = 2
    alpha_ini = [0.4, 0.6]  # 第一个为女生模型，第二个为男生模型
    mu_ini = [165, 180]
    sigma_ini = [1, 2]
    # 读取样本数据
    data = data_processing_csv('./height_data.csv')

    # 运行em算法求解高斯混合模型参数
    [alpha, mu, sigma] = gmm_em(data, k, alpha_ini, mu_ini, sigma_ini, 1e-6, 1000)
    print('\n--------------------------------')
    print("**高斯混合模型 真实参数")
    print("      均值 | 标准差 | 模型比例")
    print("女生:", 164, ' | ', 3, ' | ', 0.25)
    print("男生:", 176, ' | ', 5, ' | ', 0.75)
    print("\n**高斯混合模型 初始参数")
    print("      均值 | 标准差 | 模型比例")
    print("女生:", round(mu_ini[0], 2), ' | ', round(sigma_ini[0], 2), ' | ', round(alpha_ini[0], 2))
    print("男生:", round(mu_ini[1], 2), ' | ', round(sigma_ini[1], 2), ' | ', round(alpha_ini[1], 2))
    print("\n**高斯混合模型 估计参数")
    print("      均值 | 标准差 | 模型比例")
    print("女生:", round(mu[0], 2), ' | ', round(sigma[0], 2), ' | ', round(alpha[0], 2))
    print("男生:", round(mu[1], 2), ' | ', round(sigma[1], 2), ' | ', round(alpha[1], 2))

    # 预测每个数据点的聚类(0为女生，1为男生)
    result = np.argmax(np.array([alpha[j] * norm.pdf(data, mu[j], sigma[j]) for j in range(k)]).T, axis=1)
    gril = str(result.tolist()).count("0")
    boy = str(result.tolist()).count("1")
    print('-------------------------------')
    print("**预测结果")
    print("女生人数:", gril, "男生人数:", boy)
    print("性别比例(女:男) =", "1:{:.2f}".format(boy / gril))

    # 评估模型准确率
    gender = [0] * 500 + [1] * 1500
    accuracy = np.mean(result == gender)
    print('*准确率:', accuracy)
    print('--------------------------------')

    # 画图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(1)
    x = np.arange(len(data))
    plt.plot(x, gender, 'r-', alpha=0.8, linewidth=2, label='样本数据')
    plt.scatter(x, result, c='blue', s=3, label='模型分类结果')
    plt.legend(loc='best')
    plt.xlabel('Number', fontdict={'size': 14})
    plt.ylabel('性别(0女1男)', fontdict={'size': 14})
    plt.title("性别分类结果", fontdict={'size': 16})
    plt.savefig("./result/性别分类结果.png", dpi=600)
    # plt.show()

    plt.figure(2)
    data1 = np.random.normal(mu[0], sigma[0], int(alpha[0] * len(data)))
    data2 = np.random.normal(mu[1], sigma[1], int(alpha[1] * len(data)))
    data_fore = np.concatenate((data1, data2), axis=0)
    plt.hist(data, bins=40, histtype='bar', rwidth=0.8, label='样本数据')
    plt.hist(data_fore, bins=40, histtype='step', rwidth=0.8, label='拟合模型')
    plt.legend(loc='best')
    plt.xlabel('身高 (cm)', fontdict={'size': 14})
    plt.ylabel('数量', fontdict={'size': 14})
    plt.title("模型拟合结果", fontdict={'size': 16})
    plt.savefig("./result/模型拟合结果.png", dpi=600)
    plt.show()
