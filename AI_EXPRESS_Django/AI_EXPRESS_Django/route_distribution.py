from math import inf, sqrt
from copy import deepcopy
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pyecharts.charts import Geo
from pyecharts import options as opts
from pyecharts.globals import GeoType
import pickle
import random
from collections import defaultdict
import json


class Router(object):
    '''
    有两种分配货车的方法分别是 'min_trucks_num', 'full_trucks'
    min_trucks_num 表示局部使用最少辆的车
    full_trucks 表示使得车装得尽可能满，可能会用更多的车
    '''
    def __init__(self, method):
        # 判断使用的方法 'min_trucks_num', 'full_trucks'
        # min_trucks_num 表示局部使用最少辆的车
        # full_trucks 表示使得车装得尽可能满，可能会用更多的车
        self.method = method
        '''
        初始化无向图以及对应的数据
        '''
        self.init_data = {
            0: [2, 3, 4, 6, 1, 11],
            1: [0, 8, 9],
            2: [3, 0, 11, 12, 15],
            3: [4, 0, 2, 16],
            4: [3, 0, 6],
            5: [7, 8, 6],
            6: [5, 4, 0],
            7: [5],
            8: [5, 1],
            9: [1, 10],
            10: [9, 11, 12, 13],
            11: [0, 2, 12, 10],
            12: [2, 11, 10, 13],
            13: [10, 12, 14],
            14: [13, 15],
            15: [2, 14],
            16: [3],
        }
        self.map_dict = {}
        self.check_dict = {}
        self.m_list = ['郑州', '开封', '平顶山', '洛阳', '焦作',
                       '鹤壁', '新乡', '安阳', '濮阳', '商丘', '周口',
                       '许昌', '漯河', '驻马店', '信阳', '南阳', '三门峡']
        self.content = '''河南省,郑州,113.65,34.76
河南省,开封,114.35,34.79
河南省,平顶山,113.29,33.75
河南省,洛阳,112.44,34.7
河南省,焦作,113.21,35.24
河南省,鹤壁,114.17,35.9
河南省,新乡,113.85,35.31
河南省,安阳,114.35,36.1
河南省,濮阳,114.98,35.71
河南省,商丘,115.65,34.44
河南省,周口,114.63,33.63
河南省,许昌,113.81,34.02
河南省,漯河,114.02,33.56
河南省,驻马店,114.02,32.98
河南省,信阳,114.08,32.13
河南省,南阳,112.53,33.01
河南省,三门峡,111.19,34.76
'''

    def _initialize(self):
        '''
        初始化需要计算最短路径的数据
        '''
        for line in self.content.split('\n'):
            if line:
                _, city, x_1, x_2 = line.split(',')
                index = self.m_list.index(city)
                self.map_dict[index] = (float(x_1), float(x_2))

        # 计算欧式距离
        distance = lambda x1, x2, y1, y2: sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        # 初始化邻接表 <- 对应的位置为顶点之间的权重
        for key, value in self.init_data.items():
            self.check_dict[key] = list()
            for v in value:
                d = distance(self.map_dict[key][0], self.map_dict[v][0],
                             self.map_dict[key][1], self.map_dict[v][1])
                self.check_dict[key].append(d)

    def path(self, graph, _class, start_=0, end_=12, save=False, save_name='path.txt'):
        '''寻找两点之间的最短路径
        graph 给定一个无向图
        start_:起始点
        end_:终点
        '''
        visited = {i: False for i in range(len(self.m_list))}
        self.min_sum = inf
        self.seq_path = list()
        self.real_path = list()

        def dfs(from_, to_, edg_weight_sum):
            '''深度优先遍历无向图
            如果下一个点是终点 return
            如果下一个结点是已经遍历过的点 continue
            如果加上下一个点的边权重和大于当前最小边权重和 continue
            '''
            # 剪枝
            # global min_sum, real_path
            if to_ == end_:
                if self.min_sum > edg_weight_sum:
                    self.min_sum = edg_weight_sum
                    # print(self.seq_path, self.min_sum)
                    self.real_path = deepcopy(self.seq_path)
                return

            if all(visited.values()):
                return

            for inner_index, _node in enumerate(graph[to_]):
                if not visited.get(_node):
                    visited[_node] = True
                    self.seq_path.append(_node)
                    w = self.check_dict[to_][inner_index]
                    dfs(to_, _node, w + edg_weight_sum)
                    visited[_node] = False
                    self.seq_path.remove(_node)

        first_loop = graph[start_]
        visited[start_] = True
        self.seq_path.append(start_)

        for _index, node in enumerate(first_loop):
            visited[node] = True
            self.seq_path.append(node)
            weight = self.check_dict[start_][_index]
            dfs(start_, node, weight)  # node 是要去的结点 (0, 7, w)
            visited[node] = False
            self.seq_path.remove(node)

        if save:
            with open(save_name, 'w', encoding='utf-8') as ff:
                for k, city_index in enumerate(self.real_path):
                    c = self.m_list[city_index]
                    ff.write(c + ',') if k != len(self.real_path) - 1 else ff.write(c)

    def orginal_cluster(self, show_figure_matplotlib=False,
                        show_figure_echarts=False, test=True) -> dict:
        '''查看原始数据的聚类效果
        其表明，聚类可以大致找出运输的大方向
        show_figure:True 查看聚类的可视化
        '''
        # 连接mysql 'mysql+pymysql://用户名:密码@localhost:3306/数据库名
        engine = create_engine('mysql+pymysql://root:12345678@39.96.165.58:3306/vehicle')
        order = pd.read_sql_table('goods_order', engine)
        self.order = order.copy()
        self.engine = engine
        # order.columns = ['id', '1', '2', 'from',
        #                  '3', '4', 'to', 'weight']
        order = order[['sender_address', 'receive_address', 'order_weight']]

        # 删除起始点，对其他要分发的点进行聚类
        city = order['sender_address'][0]

        del_index = self.m_list.index(city.strip('市'))
        self.city_start = del_index  # start_city<index> -> map_dict[start_city]

        arr = [[item[0], item[1]] for item
               in self.map_dict.values()]
        data = pd.DataFrame(arr, columns=['x', 'y'])
        self.y_mean = data['y'].mean()

        # todo 测试
        if test:
            return

        # 删除起始点 by 索引
        data.drop(index=del_index, inplace=True)
        # data = data.reset_index()  # 重置索引
        print(data)
        n_cluster = 3
        km = KMeans(n_clusters=n_cluster).fit(data)
        y_pred_ = km.labels_  # 类别预测
        centroid = km.cluster_centers_  # 聚类中心
        # 标记簇标签
        city_class = {i: y_pred_[j] for i, j in zip(data.index, range(0, 16))}
        print(city_class)
        print(city_class.values())
        print(self.map_dict)

        if show_figure_matplotlib:
            color = ['red', 'blue', 'green']
            fig, ax1 = plt.subplots(1)
            y_pred = km.labels_

            # 画出对应分类的散点图
            for i in range(n_cluster):
                ax1.scatter(data.iloc[y_pred_ == i, 0], data.iloc[y_pred_ == i, 1]
                            , marker='o'
                            , s=50
                            , c=color[i]
                            )
            if n_cluster == 3:
                print(y_pred)
            # 画出聚类中心
            ax1.scatter(centroid[:, 0], centroid[:, 1]
                        , marker='x'
                        , s=50
                        , c='black')

            # 加入起始点，均值的分割线
            xx = data['x'].mean()
            yy = data['y'].mean()
            ax1.scatter(xx, yy)
            plt.hlines(yy, 111, 116, colors="black", linestyles="dashed")
            plt.show()

        if show_figure_echarts:
            print('这里没必要写了！嘻嘻！[]~(￣▽￣)~*')

        return city_class

    def _cluster(self, up_2, class_upper, plot_matplotlib=False,
                 plot_echarts=False) -> (list, list, list):
        '''对某个部分进行聚类
        class_upper:
            表示是否是在所有点的 y 均值之上的点
            key: city_index  value: 0 or 1

        :return 聚好的2类城市的列表 class_0, class_1 另一边的城市列表 (都是不包含起始点的)
        '''
        if up_2:
            # 找出上面的点进行聚类
            cities = [key for key, value in class_upper.items() if value == 1]
        else:
            cities = [key for key, value in class_upper.items() if value == 0]

        try:
            cities.remove(self.city_start)  # 去除起始点
        except:
            pass

        # todo 将与此部分的点没有联通，但是与另一部分的点有联通的归为另一边(这些点会影响聚类结果，并且不符合业务)
        should_cluster = deepcopy(self.init_data)  # 新的要聚两类中的点组成的无向图
        should_not_cluster = dict()
        other_cities = []  # 另一边的城市

        # 分开两边的点
        for k, v in class_upper.items():
            if k not in cities:
                other_cities.append(k)
                should_cluster.pop(k)
                should_not_cluster[k] = v

        # 如果一个点的!!所有子结点!!都不在这一部分里，在无向图中删除，并放到另一边
        for k, v in list(should_cluster.items()):
            not_in_sum = 0
            for next_node in v:
                if next_node not in cities:
                    not_in_sum += 1

            if not_in_sum == len(v):
                other_cities.append(k)
                cities.remove(k)
                should_cluster.pop(k)
                should_not_cluster[k] = v

        arr = []

        # 构建二维的数据矩阵
        for c in cities:
            arr.append([*self.map_dict[c]])
        data = pd.DataFrame(arr)

        n_cluster = 2
        # km = DBSCAN(eps=1.2, min_samples=3).fit(data)
        km = KMeans(n_clusters=n_cluster, random_state=6).fit(data)
        y_pred_ = km.labels_  # 类别预测

        print(cities)
        print(y_pred_)
        print(other_cities)

        centroid = km.cluster_centers_  # 聚类中心
        # city_class = {i:y_pred_[j] for i, j in zip(data.index, range(0, len(cities)))}

        # todo 将与自己的类里面无联通，但是与另一个类的点有联通的归为那一类
        class_1 = [cities[index] for index, j in enumerate(y_pred_) if j == 1]
        class_0 = [cities[index] for index, j in enumerate(y_pred_) if j == 0]

        for c in cities:
            _sum = 0
            out_sum = 0
            if c in class_0:
                c_mark = 0  # c_mark 当前点的类别
            else:
                c_mark = 1
            c_list = should_cluster[c]  # 单点点对应的子结点列表
            for node in c_list:
                if node not in cities:
                    out_sum += 1
                    continue
                node_mark = 0 if node in class_0 else 1
                if c_mark != node_mark:
                    _sum += 1
            if len(c_list) == _sum + out_sum:
                if c_mark == 0:
                    class_0.remove(c)
                    class_1.append(c)
                else:
                    class_1.remove(c)
                    class_0.append(c)

        if plot_matplotlib:
            color = ['red', 'blue']
            fig, ax1 = plt.subplots(1)
            y_pred = km.labels_

            for i in range(n_cluster):
                ax1.scatter(data.iloc[y_pred == i, 0], data.iloc[y_pred == i, 1]
                            , marker='o'
                            , s=50
                            , c=color[i]
                            )

                ax1.scatter(centroid[:, 0], centroid[:, 1]
                            , marker='x'
                            , s=50
                            , c='black'
                            )
            plt.savefig('./ttesst.jpg')
            plt.show()

        # 不需要聚类那一边的点
        # class_another = [x for x in range(0, 17) if x not in class_0
        #                     and x not in class_1].remove(self.city_start) 注意这个列表推导式有毒！

        # 另一边的点
        class_another = []
        for x in range(0, 17):
            if x not in class_0:
                if x not in class_1:
                    class_another.append(x)

        # fp =  open('./地图可视化.pk', 'wb')
        # pickle.dump({2:class_0, 5: class_1, 8:class_another}, fp)
        # f = open('./城市经纬度.pk', 'wb')
        # pickle.dump(self.map_dict, f)
        # ff = open('./m_list.pk', 'wb')
        # pickle.dump(self.m_list, ff)
        # fp.close()
        # f.close()
        # ff.close()

        if plot_echarts:
            temp_d = {}
            d = {2: class_0, 5: class_1, 8: class_another}
            for key, value in d.items():
                for v in value:
                    if v == self.city_start:
                        continue
                    temp_d[self.m_list[v]] = key
            temp_d[self.m_list[self.city_start]] = 11
            g = Geo()
            g.add_schema(maptype='河南')
            data_pair = list(temp_d.items())
            g.add('', data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=15)
            g.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            pieces = [
                {'min': 0, 'max': 3, 'label': '类别1', 'color': '#50A3BA'},  # blue
                {'min': 4, 'max': 6, 'label': '类别2', 'color': '#81AE9F'}, # green
                {'min': 7, 'max': 8, 'label': '类别3', 'color': '#E2C568'}, # yellow
                {'min': 10, 'max': 12, 'label': '起始点', 'color': '#FF8C00'} # orange
            ]
            g.set_global_opts(
                visualmap_opts=opts.VisualMapOpts(is_piecewise=True, pieces=pieces),
                title_opts=opts.TitleOpts(title=f"物流路径规划-城市聚类-起始点{self.m_list[self.city_start]}"),
            )

            g.render(f'物流路径规划之城市聚类-{self.m_list[self.city_start]}.html')
            # self.fix_cluster_graph(class_0, class_1, class_another)

        return class_0, class_1, class_another

    def fix_cluster_graph(self, class1, class2, class3):
        '''
        判断某个类是否于起始点连通
        如果与起始点不连通，则寻找从起始点到这个类的最短路径
        用一个全局变量记录其中的结点，后面分配重量的时候使用
        '''
        class_list = [class1, class2, class3]
        # 对于最上面的点和最下面的点会出现空列表 []
        class_list = [i for i in class_list if i!=[self.city_start]]
        print('class_list -> ', class_list)
        is_connected = {}  # <dict> key: class_list_index  value: bool

        # 判断每个类与起始点是否连通
        for index, class_i in enumerate(class_list):
            for node in class_i:
                node_list = self.init_data[node]
                for n in node_list:
                    if n == self.city_start:
                        is_connected[index] = True
                        break
            if not is_connected.get(index):
                is_connected[index] = False

        # 对于与起始点没有连通的那一个簇, 添加起始点到这个簇里面某一个点中间经过的结点
        # 用一个全局变量标记这些 "中间点"
        # 正常来说起始点最多只会与一个类不存在连通,等后边每一个都验证一遍就知道了
        for key, value in is_connected.items():
            if not value:
                extend_list = []
                class_j = class_list[key]  # 与起始点没有连通的那一个簇的点
                min_path_weight = inf  # 记录最小值
                prior = None  # 记录最小值对应的路径经过的结点
                for end in class_j:
                    self.path(self.init_data, class_j, start_=self.city_start, end_=end)
                    if self.min_sum < min_path_weight:
                        min_path_weight = self.min_sum
                        prior = self.real_path
                self.prior = prior
                for node in prior:
                    if node not in class_j:
                        if node != self.city_start:
                            extend_list.append(node)
                extend_list = [self.city_start, *extend_list, *class_j]
                class_list[key] = extend_list

        self.is_connected = is_connected

        return class_list

    def get_best_trucks(self, weight_sum):
        '''
        深度优先搜索找到最佳的
        找到：能装满，并且车的载重量之和最小的那几辆车去运输
        寻找每一次路径的局部最优解
        :return: truck_list, truck_id_list
        '''
        capacity_list = []
        id_list = []
        self.real_capacity_list = []
        self.real_id_list = []
        self.min_sum = inf

        def insert(index):
            capacity_list.append(self.truck_list_all[index])
            id_list.append(self.truck_id_list_all[index])

        def remove(index):
            capacity_list.remove(self.truck_list_all[index])
            id_list.remove(self.truck_id_list_all[index])

        def dfs():
            if sum(capacity_list) > weight_sum:
                if sum(capacity_list) < self.min_sum:
                    self.real_capacity_list = deepcopy(capacity_list)
                    self.real_id_list = deepcopy(id_list)
                    self.min_sum = sum(capacity_list)
                    return
                return

            for j in range(len(self.truck_list_all)):
                if not visited[j]:
                    visited[j] = True
                    insert(j)
                    dfs()
                    remove(j)
                    visited[j] = False

        visited = {i: False for i in range(len(self.truck_list_all))}
        for i in range(len(self.truck_list_all)):
            if not visited[i]:
                visited[i] = True
                insert(i)
                dfs()
                remove(i)
                visited[i] = False

        # return real_capacity_list, real_id_list


    def calculate(self):
        '''
        分发重量到每一个点
        求每一个点之间的最短路径，如果前面是交集的，后面的点就从那里走
        '''
        order = self.order
        truck = pd.read_sql_table('truck', self.engine)
        # truck['truck_id'] = truck['truck_id'] - 1

        def apply_func(arg):
            arg = arg.strip('市')
            here = self.m_list.index(arg)
            return here

        truck['carrying_capacity'] = truck['carrying_capacity'] * 1000  # 吨和 kg 单位换算

        # 初始化要进行货车分配的变量
        self.truck_list_all = []
        self.truck_id_list_all = []

        for row in truck.iterrows():
            if row[1]['truck_status'] == 0:
                self.truck_list_all.append(row[1]['carrying_capacity'])
                self.truck_id_list_all.append(row[1]['truck_id'])



        if truck['carrying_capacity'].sum() < order['order_weight'].sum():
            print('这车运不完呀老弟 Σ( ° △ °|||)︴')

        capacity_sort = truck['carrying_capacity'].sort_values(ascending=False)  # 根据车的载重量进行排序 <pandas.Series>
        print('orginal -> ', capacity_sort)
        capacity_sort_stb = truck['carrying_capacity'].sort_values()
        order['receive_address'] = order['receive_address'].apply(apply_func)
        city_weight = order.groupby(by='receive_address')
        city_weight_sum = city_weight.sum()
        city_weight_sum = city_weight_sum['order_weight']  # 分组聚合后是二重索引，分组键为一级索引

        # todo 找到要去的每一个类里面
        class_list = self.distribute()  # (list, list , list) 0 1 another, or just 0 and 1

        # ()
        memory_seq = []
        fix_class_list = self.fix_cluster_graph(*class_list)

        # 找对短路径的时候不能回到起始点 self.start_city
        for index, _class in enumerate(class_list):
            # 切断与其他类的连通关系，但是与起始点要保持，寻找里面每一个点的最短路径
            graph = {}  # 类和起始点组成的无向图
            for c in _class:
                graph[c] = self.init_data[c]

            graph[self.city_start] = self.init_data[self.city_start]  # 添加起始点的连通状态
            temp = deepcopy(_class)
            temp.append(self.city_start)

            # 删除和其他类的点之间的联通关系
            for t in temp:
                node_list = graph[t]
                new_node_list = list()
                for node in node_list:
                    if node in temp:
                        new_node_list.append(node)
                        graph[t] = new_node_list
            # 如果起始点与某个类之间没有连通
            # 向当前这个图中添加起始点到这个类最短路径上的结点，并更新连通关系
            try:
                if not self.is_connected[index]:
                    d = len(self.prior) - 2
                    if d == 1:
                        graph[self.prior[-1]].append(self.prior[-2])
                        graph[self.prior[0]] = [self.prior[1]]
                        graph[self.prior[1]] = [self.prior[0], self.prior[-1]]
                    elif d > 1:
                        graph[self.prior[0]] = [self.prior[1]]
                        graph[self.prior[-1]].append(self.prior[-2])
                        for inner_index, j in enumerate(self.prior[1:-1]):
                            graph[j] = [self.prior[inner_index], self.prior[inner_index+2]]
            except KeyError:
                pass

            memory_seq.append((graph, _class))

        print(memory_seq)

        get_weight = {key:False for key in range(0, 17)}  # 标记某个结点是否分配过重量
        json_dict = {}  # 最后得到的结果
        global_num = 0

        # todo 寻找起点到每一个结点的最短路径经过的结点
        for graph, _class in memory_seq:
            all_path = []
            for end in _class:
                if end == self.city_start:
                    continue
                self.path(graph, _class, start_=self.city_start, end_=end)
                all_path.append(self.real_path)
            # print('all_path->', all_path)

            # 对于一个路径经过的结点组成的数组，如果里面的全部元素是另一个数组的子数组(即子路径)，那么就属于同一条路
            # todo 最短路径前 k 个结点的合并子集
            # 判断 1. 全部在里面
            # 判断 2. 前面 n 个元素在里面，第 n+1 不等
            sub_set = {index: row for index, row in enumerate(all_path)}
            sub_set_copy = deepcopy(sub_set)

            for index_i, row_i in sub_set.items():
                for index_j, row_j in sub_set.items():
                    if index_i == index_j:
                        continue

                    if len(row_i) < len(row_j):
                        flag = 1
                        for i in range(len(row_i)):
                            if row_i[i] != row_j[i]:
                                flag = 0
                        if flag:
                            # print(index_i, index_j)
                            sub_set_copy.pop(index_i)
                            break

            # print('go---->', sub_set_copy)

            # 计算每个类中，总订单重量
            # 如果某个城市订单总量为 0
            for path in sub_set_copy.values():
                print(path)
                path.remove(self.city_start)
                weight_sum = 0  # 这条路径上的订单总重量
                for c in path:
                    if not get_weight[c]:
                        weight = city_weight_sum[c]
                        weight_sum += weight
                        get_weight[c] = True

                # print('weight_sum -> ', weight_sum)

                truck_list = list()
                truck_id_list = []
                temp_weight = weight_sum

                # todo 分配
                if self.method == 'full_trucks':
                    self.get_best_trucks(temp_weight)
                    truck_list = deepcopy(self.real_capacity_list)
                    truck_id_list = deepcopy(self.real_id_list)
                else:
                    def delete(Index):
                        '''根据值，找到 Series 的索引，删除 capacity_sort 的对应行'''
                        print('Index', Index)
                        capacity_sort_stb.drop(index=Index, inplace=True)
                        capacity_sort.drop(index=Index, inplace=True)
                        truck_id_list.append(Index)

                    # 删的时候 capacity_sort_stb 和 capacity_sort 对应的索引一起删
                    # 如果 temp_weight < 核载最大的车，找核载大于temp_weight，但是核载最小的车
                    if capacity_sort.iloc[0] > temp_weight:
                        for r in capacity_sort_stb.items():
                            if r[1] > temp_weight:
                                if int(truck.iloc[r[0]]['truck_status']) == 0:
                                    truck_list.append(r[1])
                                    delete(r[0])
                                    break
                    else:  # 最大的核载量比 temp_weight 都小
                        # while temp_weight > capacity_sort.max():
                        for r in capacity_sort.items():
                            if temp_weight > capacity_sort.max():
                                if int(truck.iloc[r[0]]['truck_status']) == 0:
                                    temp_weight -= r[1]
                                    truck_list.append(r[1])
                                    delete(r[0])
                                    if temp_weight < capacity_sort.min():
                                        break
                            else:
                                # 判断一下是不是每一个值都相同
                                if len(set(capacity_sort)) == 1:
                                    pass
                                else:
                                    for k in range(1, capacity_sort.shape[0]):
                                        if temp_weight < capacity_sort.iloc[k - 1] and \
                                                temp_weight > capacity_sort.iloc[k]:

                                            t_capacity = capacity_sort.iloc[k - 1]
                                            for x, y in capacity_sort.items():
                                                if y == t_capacity:
                                                    break
                                            if truck.iloc[x]['truck_status'] == 0:
                                                flag = 1
                                                truck_list.append(t_capacity)
                                                delete(x)
                                                temp_weight -= t_capacity

                                    if flag == 1:
                                        break
                                if flag == 1:
                                    break

                        if temp_weight > 0:
                            assert capacity_sort.shape[0] > 0
                            for r in capacity_sort_stb.items():
                                if r[1] > temp_weight:
                                    if int(truck.iloc[r[0]]['truck_status']) == 0:
                                        truck_list.append(r[1])
                                        delete(r[0])
                                        break

                # print(self.real_capacity_list)
                # print(self.real_id_list)
                print('truck_id_list -> ', truck_id_list)
                print('weight_sum -> ', weight_sum)
                print('truck_list -> ', truck_list)

                # 找出所有的这条路上的订单信息
                all_order = []
                s = 0
                for r in order.iterrows():
                    # todo 后面整
                    if r[1]['order_status'] == 0:
                        if r[1]['receive_address'] in path:
                            all_order.append(r[1]['order_id'])
                            s += r[1]['order_weight']
                # print('gg_list', all_order)
                print('gg!!!s----->', s)

                # 给不同重量的车分配订单
                truck_index = 0
                ssum = 0
                truck_orders_dict = dict()
                orders_weight_dict = defaultdict(list)

                for _index, g in enumerate(all_order):
                    r_weight = order[order['order_id'] == g]['order_weight'].values[0]
                    # print(r_weight)
                    if r_weight + ssum > truck_list[truck_index]:
                        truck_orders_dict[truck_index] = ssum
                        truck_index += 1
                        ssum = 0
                    # 如果为最后一条，把剩下的全部加进去
                    if truck_index == len(truck_list) - 1:
                        sssum = 0
                        for jj in range(_index, len(all_order)):
                            sssum += order[order['order_id'] == all_order[jj]]['order_weight'].values[0]
                            orders_weight_dict[truck_index].append(all_order[jj])
                        truck_orders_dict[truck_index] = sssum
                        break
                    ssum += r_weight
                    orders_weight_dict[truck_index].append(g)

                # print(path)
                # print(truck_orders_dict) # truck_orders_dict 和 orders_weight_dict 是相反的
                # print(truck_list)
                # print(orders_weight_dict)
                json_dict[global_num] = {}
                json_dict[global_num]['truck_number_list'] = truck_id_list
                json_dict[global_num]['path'] = path
                json_dict[global_num]['orders_weight'] = [truck_orders_dict[i] for i in range(len(truck_list))]
                json_dict[global_num]['order_id_list'] = [orders_weight_dict[j] for j in range(len(truck_list))]
                global_num += 1

        with open('./json_data.json', 'w') as fp:
            json.dump(json_dict, fp)
        return json_dict



    def distribute(self):
        '''
        y_0 起始点的 y 值
        在起始点 y_0 之上  class_upper.value = 1
        up_2: True  要对位于起始点之上的点聚2类
        '''
        # city_class = self._cluster()
        x_0, y_0 = self.map_dict.get(self.city_start)  # 起始点
        # 根据起始点 以下分为一部分，以上分为另一部分
        class_upper = dict()
        for key, value in self.map_dict.items():
            _, y = value
            if y > y_0:
                class_upper[key] = 1
            else:
                class_upper[key] = 0

        # 起始点在均值之上的话下面的点聚2个类，否则下面聚2个类
        # 聚2个类就是为了找到2个大致要运输的方向
        up_2 = True  # up_2 = True 表示上面要聚类两个
        if y_0 > self.y_mean:
            up_2 = False

        class_0, class_1, class_another = self._cluster(up_2, class_upper, plot_echarts=True, plot_matplotlib=False)

        return class_0, class_1, class_another

    def run(self):
        self._initialize()
        # self.path()
        self.orginal_cluster()
        # self.distribute()
        j = self.calculate()
        return j


if __name__ == '__main__':
    r = Router(method='full_trucks')
    # r = Router(method='min_trucks_sum')
    r.run()