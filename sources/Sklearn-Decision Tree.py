# -*- coding: UTF-8 -*-
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus

if __name__ == '__main__':
    # 加载文件
    with open('test.txt', 'r') as fr:
        # 处理文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]

    # 提取每组数据的类别，保存在列表里
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    # print(lenses_target)

    # 特征标签
    lensesLabels = ['age', 'diligent', 'Recharge', 'lucky']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    # 提取信息，生成字典
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)
    # 生成pandas.DataFrame
    lenses_pd = pd.DataFrame(lenses_dict)
    # print(lenses_pd)

    # 创建LabelEncoder()对象，用于序列化
    le = LabelEncoder()

    # 序列化pandas数据
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)

    # 创建DecisionTreeClassifier()类,决策树的最大深度，默认不设置，但如果特征过多，建议设置
    clf = tree.DecisionTreeClassifier(max_depth=4)

    # 使用数据，构建决策树
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)

    # 经常被用来作字符串的缓存，它的部分接口跟文件一样，可以
    # 认为是作为"内存文件对象"，简而言之，就是为了方便
    dot_data = StringIO()
    # 绘制决策树
    # out_file:输出文件的句柄或名称
    # feature_names:每个功能的名称
    # class_names:每个目标类的名称按升序排列。仅与分类相关且不支持多输出。如果True，则显示类名的符号表示。
    # filled:设置为时True，绘制节点以指示用于分类的多数类，用于回归的极值，或用于多输出的节点的纯度。
    # rounded:
    # special_characters:
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # 保存绘制好的决策树，以PDF的形式存储。
    graph.write_pdf("tree.pdf")

    print(clf.predict([[1, 1, 1, 0]]))  # 预测
