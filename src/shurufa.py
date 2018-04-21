import numpy as np

def get_hanzi_matrix():
    matrix = open('../data/data.txt', 'r')
    hanzi_matrix = np.zeros([7272, 7272])
    for line in matrix:
        row = line.split()
        row_num = int(row[0])
        for i in range(int((len(row)-2)/2)):
            hanzi_matrix[row_num][int(row[2*i+1])]=int(row[2*i+2])
    matrix.close()
    return hanzi_matrix

py2hanzi = np.load('../data/py2hanzi.npy').item()
hanzi_dict = np.load('../data/hanzi_dict.npy').item() # 汉字与编码映射表
hanzi_matrix = get_hanzi_matrix()
# hanzi_matrix = np.load('hanzi_matrix.npy') # 词语矩阵
hanzi_num = np.load('../data/hanzi_num.npy') # 汉字出现次数
total_num = sum(hanzi_num)

class GraphNode(object): # 有向图节点
    def __init__(self, hanzi, count, express):
        # 当前节点所代表的汉字（即状态）
        self.hanzi = hanzi
        # 当前汉字的出现次数
        self.count = count
        # 当前汉字与其他汉字共同出现的次数（词组数列）
        self.express = express
        # 最优路径时，从起点到该节点的最高分
        self.max_prob = 0.0
        # 最优路径时，该节点的前一个节点，用来输出路径的时候使用
        self.prev_node = None

class GraphLevel(object): # 有向图中的图层
    def __init__(self):
        self.level = []

    def append(self, node):
        # 添加一个节点
        self.level.append(node)

class BestPath(object): # 当前最佳路径
    def __init_(self):
        self.prob = 0.0
        self.prev = None
        self.next = None
        self.level = 0 # 当前层数

class Graph(object): # 有向图
    def __init__(self, pinyins):
        """根据拼音所对应的所有汉字组合，构造有向图"""
        self.levels = []
        for py in pinyins:
            level = []
            # 从拼音、汉字的映射表中读取汉字的出现次数以及汉字的词组数列
            hanzi_list = py2hanzi[py]
            for hanzi in hanzi_list:
                code = hanzi_dict[hanzi]
                count = hanzi_num[code]
                express = hanzi_matrix[code][:]
                node = GraphNode(hanzi, count, express)
                level.append(node)
            self.levels.append(level)


def viterbi(graph, lamda):
    def viterbi_i(i, graph):
        # 对于有向图，在第i层求所有节点的到该节点的最大概率
        if i == 0: # 如果为第0层
            for node_j in graph.levels[i]:
                code_j = hanzi_dict[node_j.hanzi] # i层j节点的编码
                num_j = hanzi_num[code_j]
                node_j.max_prob = num_j / total_num
            return

        for node_j in graph.levels[i]:
            # 对于第j个节点，需要与前面第i-1层的所有节点匹配，求最大概率
            probs =  []
            code_j = hanzi_dict[node_j.hanzi] # i层j节点的编码
            num_j = hanzi_num[code_j]

            for node_k in graph.levels[i-1]: # 对于第i-1层的k节点
                code_k = hanzi_dict[node_k.hanzi] # 上一层节点的编码
                num_k = hanzi_num[code_k]
                P_emission = lamda * node_k.express[code_j] / num_k + (1-lamda) * num_k / total_num # 发射概率，已经平滑
                probs.append(node_k.max_prob * P_emission)

            # 获取最大概率在i-1层的位置
            max_k = probs.index(max(probs))
            node_j.max_prob = probs[max_k]
            node_j.prev_node = graph.levels[i-1][max_k]
        return
    level_len = len(graph.levels)
    for i in range(level_len):
        viterbi_i(i, graph)


def bestpath(graph):
    level_len = len(graph.levels)
    max_prob = []
    for node in graph.levels[level_len-1]:
        max_prob.append(node.max_prob)
    max_index = max_prob.index(max(max_prob))
    node = graph.levels[level_len-1][max_index]
    result = []
    real_result = ''
    while True:
        result.append(node.hanzi)
        node = node.prev_node
        if node is None:
            break
        if node.prev_node is None:
            result.append(node.hanzi)
            break
    while len(result) > 0:
        hz = result.pop()
        real_result+=hz
        # print(hz, end='')
    # print('\n')
    return real_result

def get_accuracy(s, s_true):
    n = len(s)
    count = 0
    for x, y in zip(s, s_true):
        if x == y:
            count+=1
    return count / n

def par_sel(lamda):
    for lamda in [lamda]:
        print(lamda)
        accuracy = []
        count_line = 0
        for line in input_py:
            pinyins = line.lower().split()
            new_pys = []
            flag = True
            for py in pinyins:
                if py == 'nv':
                    new_pys.append('nü')
                elif py == 'lv':
                    new_pys.append('lü')
                elif py == 'qv':
                    new_pys.append('qu')
                elif py == 'xv':
                    new_pys.append('xu')
                # elif py==''
                else:
                    new_pys.append(py)
            mygraph = Graph(new_pys)
            bps = []
            viterbi(mygraph, lamda)
            result = bestpath(mygraph)
            output_hz.write('%s\n' % result)


if __name__ == '__main__':
    input_py = open('../data/input.txt', 'r')
    output_hz = open('../data/output.txt', 'w')
    par_sel(0.99999)
