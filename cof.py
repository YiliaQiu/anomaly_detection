# https://zhuanlan.zhihu.com/p/362358580
import numpy as np
from pyod.models.cof import COF
cof = COF(contamination = 0.06,  ## 异常值所占的比例
          n_neighbors = 20,      ## 近邻数量
        )
cof_label = cof.fit_predict(iris.values) # 鸢尾花数据
print("检测出的异常值数量为:",np.sum(cof_label == 1))