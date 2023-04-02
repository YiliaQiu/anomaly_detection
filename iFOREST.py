# IsolationForest()不允许缺失值
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option("display.max_columns",None)
data = load_iris(as_frame=True)
X,y=data.data,data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)



iforest = IsolationForest(n_estimators=20, contamination=0.1, random_state=0)
iforest.fit_predict(X)
iforest_score = iforest.decision_function(X_train)
print("iforest_score={}".format(iforest_score))
print("iforest.offset_={}".format(iforest.offset_)) # 临界值

pred = iforest.predict(X_test) # decision_function<0的就是-1，其他的是1
print("pred={}".format(pred))
test_result = X_test
test_result['y_test'] = y_test
# test_result['scores'] = iforest.decision_function(X_test)
test_result['anomaly_label'] = pred
print(test_result)

# sklearn还提供其他离群值检测方法，如：MCD方法，通过covariance.EllipticEnvelope()实现
# 网上有讨论说，低维情况MCD方法较好，高维情况孤立森林方法较好。大家可自行尝试。
