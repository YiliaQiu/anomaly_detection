from sklearn.neighbors import LocalOutlierFactor as LOF
X_list = list()
X1 = [[-1.1], [0.2], [100.1], [0.3]]
X2 = [[10], [0], [-1.1], [0.3]]
X_list.append(X1)
X_list.append(X2)
for i in range(2):
    print("-"*20,"第{}个特征".format(i),"-"*20)
    clf = LOF(n_neighbors=2) # 最近邻参数设置
    res = clf.fit_predict(X_list[i])
    print(res)
    print(clf.negative_outlier_factor_)