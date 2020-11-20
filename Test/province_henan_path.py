from math import inf, sqrt
from copy import deepcopy
import sys


'''
河南省:
郑州_0, 开封_1, 平顶山_2, 洛阳_3, 焦作_4,
鹤壁_5, 新乡_6, 安阳_7, 濮阳_8, 商丘_9, 周口_10,
许昌_11, 漯河_12, 驻马店_13, 信阳_14, 南阳_15, 三门峡_16
'''

#  邻接表表示无向图
init_data = {
    0: [2, 3, 4, 6, 1, 11],
    1: [0, 8, 9],
    2: [3, 0, 11, 12, 15],
    3: [4, 0, 2, 16],
    4: [3, 0, 6],
    5: [7, 8, 6],
    6: [5, 4, 0],
    7: [5],
    8: [5, 1],
    9: [1],
    10: [11, 12, 13],
    11: [0, 2, 12, 10],
    12: [2, 11, 10, 13],
    13: [10, 12, 14],
    14: [13, 15],
    15: [2, 14],
    16: [3],
}

# 0: (x_1, x_2) (经度, 纬度)
map_dict = {}

# 检验字典 0:[dis_1, dis_2]
check_dict = {}

m_list = ['郑州', '开封', '平顶山', '洛阳', '焦作',
          '鹤壁', '新乡', '安阳', '濮阳', '商丘', '周口',
          '许昌', '漯河', '驻马店', '信阳', '南阳', '三门峡']

content = '''河南省,郑州,113.65,34.76
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

for line in content.split('\n'):
    if line:
        _, city, x_1, x_2 = line.split(',')
        index = m_list.index(city)
        map_dict[index] = (float(x_1), float(x_2))

# print(map_dict)

arg1, arg2 = sys.argv[1:]
# 起始点
start_ = m_list.index(arg1)
# 终点
end_ = m_list.index(arg2)
# 顶点数，边数
vexnum = edgnum = len(m_list)
# 计算欧式距离
distance = lambda x1, x2, y1, y2: sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 初始化邻接表 <- 对应的位置为顶点之间的权重
for key, value in init_data.items():
    check_dict[key] = list()
    for v in value:
        d = distance(map_dict[key][0], map_dict[v][0],
                               map_dict[key][1], map_dict[v][1])
        check_dict[key].append(d)


# print(check_dict)

visited = {i:False for i in range(len(m_list))}
min_sum = inf
seq_path = list()
real_path = list()


def dfs(from_, to_, edg_weight_sum):
    '''深度优先遍历无向图
    如果下一个点是终点 return
    如果下一个结点是已经遍历过的点 continue
    如果加上下一个点的边权重和大于当前最小边权重和 continue
    '''
    # 剪枝
    global min_sum, real_path
    if to_ == end_:
        # todo：记录边的权重最小值
        # second_index = init_data[from_].index(to_)
        # current_edg_weight = check_dict[from_][second_index]
        # current_weight = current_edg_weight + edg_weight_sum
        # if min_sum > current_weight:
        #     min_sum = current_weight
        #     # print(visited)
        #     print(seq_path, min_sum)
        if min_sum > edg_weight_sum:
            min_sum = edg_weight_sum
            print(seq_path, min_sum)
            real_path = deepcopy(seq_path)
        return

    if all(visited.values()):
        return

    for inner_index, _node in enumerate(init_data[to_]):
        if not visited.get(_node):
            visited[_node] = True
            seq_path.append(_node)
            w = check_dict[to_][inner_index]
            dfs(to_, _node, w + edg_weight_sum)
            visited[_node] = False
            seq_path.remove(_node)


first_loop = init_data[start_]
visited[start_] = True
seq_path.append(start_)


for _index, node in enumerate(first_loop):
    visited[node] = True
    seq_path.append(node)
    weight = check_dict[start_][_index]
    dfs(start_, node, weight)  # node 是要去的结点 (0, 7, w)
    visited[node] = False
    seq_path.remove(node)

# print(real_path)
with open('./path.txt', 'w', encoding='utf-8') as ff:
    for k, city_index in enumerate(real_path):
        c = m_list[city_index]
        ff.write(c + ',') if k != len(real_path)-1 else ff.write(c)
