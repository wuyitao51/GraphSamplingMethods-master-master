import sample
import networkx as nx
import matplotlib.pyplot as plt
import csv
import collections
import pandas as pd

# set path
nodefile_path = 'Result/Undirected/nodes/relationship_nodes.csv'
edgefile_path = 'Result/Undirected/edges/relationship_edges.csv'
classfile_path = 'Result/Undirected/classes/nodes/relationship_nodes.csv'
orig_edgefile_path = 'Result/Undirected/classes/edges/relationship_edges.csv'

# load graph to networkx
f = open("data/relationship.csv", "r")
reader1 = csv.reader(f)
edges = []
for item in reader1:
    edges.append([int(item[0]), int(item[1])])
f.close()

# Undirected
# G = nx.Graph()
# G.add_edges_from(edges)
# Directed
G = nx.DiGraph()
G.add_edges_from(edges)


# set sampling rate
total = len(G.nodes())
rate = 0.5
sample_rate = int(total * rate)

# run sample algorithm
'''
    sample1 - random_node
    sample2 - random_edge
    sample3 - bfs
    sample4 - snowball
    sample5 - forestfire
    sample6 - random_walk_sampling_simple
    sample7 - random_walk_sampling_with_fly_back
    sample8 - random_walk_induced_graph_sampling
    sample9 - mhrw
    sample10 - induced_mhrw
    sample11 - ties
    sample12 - rj
    
'''

# make an object and call function RN
# RN_object = sample.RandomNode()
# RN_sample = RN_object.randomnode(G, sample_rate)  # graph, number of nodes to sample

# make an object and call function RE
# RE_object = sample.RandomEdge()
# RE_sample = RE_object.randomedge(G, sample_rate)  # graph, number of nodes to sample

# make an object and call function BFS
# BFS_object = sample.BFS()
# BFS_sample = BFS_object.bfs(G, sample_rate)  # graph, number of nodes to sample

# make an object and call function SB
# SB_object = sample.Snowball()
# SB_sample = SB_object.snowball(G, sample_rate, 6)  # graph, number of nodes to sample

# make an object and call function FF
# FF_object = sample.ForestFire()
# FF_sample = FF_object.forestfire(G, sample_rate)  # graph, number of nodes to sample

# make an object and call function RW
# RW_object = sample.SRW_RWF_ISRW()
# RW_sample = RW_object.random_walk_sampling_simple(G, sample_rate)  # graph, number of nodes to sample

# make an object and call function RWF
# RWF_object = sample.SRW_RWF_ISRW()
# RWF_sample = RWF_object.random_walk_sampling_with_fly_back(G, sample_rate, 0.2)  # graph, number of nodes to sample

# make an object and call function ISRW
# ISRW_object = sample.SRW_RWF_ISRW()
# ISRW_sample = ISRW_object.random_walk_induced_graph_sampling(G, sample_rate)  # graph, number of nodes to sample

# make an object and call function MHRW
# MHRW_object = sample.MHRW()
# MHRW_sample = MHRW_object.mhrw(G, sample_rate)  # graph, number of n

# make an object and call function MHRW
# ISMHRW_object = sample.MHRW()
# ISMHRW_sample = ISMHRW_object.induced_mhrw(G, sample_rate)  # graph, number of n

# make an object and call function TIES
# TIES_object = sample.TIES()
# TIES_sample = TIES_object.ties(G, sample_rate)  # graph, number of n

# make an object and call function RJ
RJ_object = sample.RJ()
RJ_sample = RJ_object.rj(G, sample_rate)  # graph, number of n

# info
# print(FF_sample.nodes())
# print("Number of nodes sampled=", len(FF_sample.nodes()))
# print("Number of edges sampled=", len(FF_sample.edges()))
# print("degree", nx.degree_histogram(FF_sample))
# print("cluster",  nx.average_clustering(FF_sample))

#1、2、3分开运行
#-------------------------------------------------------------------------------------------
#1.原始图和采样图
# spring_pos = nx.spring_layout(G)
# plt.subplot(121)
# plt.title('original graph')
# nx.draw(G, spring_pos, with_labels=True)
#
# plt.subplot(122)
# plt.title('sample graph')
# nx.draw(RJ_sample, spring_pos, with_labels=True)
#-------------------------------------------------------------------------------------------
#2.度分布
# plt.subplot(221)
# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# print('Degree sequence', degree_sequence)
# plt.loglog(degree_sequence, 'b-', marker='o')
# plt.title('Degree Rank of original graph')
# plt.ylabel('degree')
# #plt.xlabel('rank')
#
# plt.subplot(222)
# degree_sequence = sorted([d for n1, d in RWF_sample.degree()], reverse=True)
# print('Degree sequence', degree_sequence)
# plt.loglog(degree_sequence, 'b-', marker='o')
# plt.title('Degree Rank of sample graph')
# plt.ylabel('degree')
# #plt.xlabel('rank')
#
# ax = plt.subplot(223)
# degree_sequence = sorted([d for n2, d in G.degree()], reverse=True)
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())
# plt.bar(deg, cnt, width=0.80, color='b')
# plt.title('Degree Histogram of original graph')
# plt.ylabel('Count')
# plt.xlabel('Degree')
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)
#
# ax = plt.subplot(224)
# degree_sequence = sorted([d for n3, d in RWF_sample.degree()], reverse=True)
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())
# plt.bar(deg, cnt, width=0.80, color='b')
# plt.title('Degree Histogram of sample graph')
# plt.ylabel('Count')
# plt.xlabel('Degree')
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)

#2-1 仅有向图--出入度分布----in_degree()    out_degree()
plt.subplot(221)
degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)
print('In-Degree sequence', degree_sequence)
plt.loglog(degree_sequence, 'b-', marker='o')
plt.title('In-Degree Rank of original graph')
plt.ylabel('in-degree')
#plt.xlabel('rank')

plt.subplot(222)
degree_sequence = sorted([d for n1, d in RJ_sample.out_degree()], reverse=True)
print('Degree sequence', degree_sequence)
plt.loglog(degree_sequence, 'b-', marker='o')
plt.title('In-Degree Rank of sample graph')
plt.ylabel('in-degree')
#plt.xlabel('rank')

ax = plt.subplot(223)
degree_sequence = sorted([d for n2, d in G.out_degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.bar(deg, cnt, width=0.80, color='b')
plt.title('In-Degree Histogram of original graph')
plt.ylabel('Count')
plt.xlabel('In-Degree')
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

ax = plt.subplot(224)
degree_sequence = sorted([d for n3, d in RJ_sample.out_degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.bar(deg, cnt, width=0.80, color='b')
plt.title('In-Degree Histogram of sample graph')
plt.ylabel('Count')
plt.xlabel('In-Degree')
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)
#-------------------------------------------------------------------------------------------
#3.新画的图--都是字典数据,改第二行的统计指标即可
#i.统计聚类系数的分布图和频率分布直方图
#ii.统计度中心度、接近中心度，中介中心度的分布图和频率分布直方图

#有向图一些度的指标也分出入--nx.in_degree_centrality和nx.out_degree_centrality

# plt.subplot(221)
# degree_clustering = sorted([v for k, v in nx.average_neighbor_degree(G).items()], reverse=True)
# print('Degree Clustering', degree_clustering)
# #plt.hist(degree_clustering, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
# #                                  0.8, 0.85, 0.9, 0.95, 1.0])
# plt.hist(degree_clustering, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# plt.title("Average Neighbor-Degree Histogram of Original Graph")
#
# plt.subplot(222)
# degree_clustering = sorted([v1 for k1, v1 in nx.average_neighbor_degree(RW_sample).items()], reverse=True)
# print('Degree Clustering', degree_clustering)
# #plt.hist(degree_clustering, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,
# #                                  0.8, 0.85, 0.9, 0.95, 1.0])
# plt.hist(degree_clustering, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# plt.title("Average Neighbor-Degree Histogram of Sample Graph")
#
# plt.subplot(223)
# degree_clustering = sorted([v2 for k2, v2 in nx.average_neighbor_degree(G).items()], reverse=True)
# plt.loglog(degree_clustering, 'b-', marker='o')
# plt.title('Average Neighbor-Degree Rank of Original Graph')
# plt.ylabel('Average Neighbor-Degree')
# plt.xlabel('rank')
#
# plt.subplot(224)
# degree_clustering = sorted([v2 for k2, v2 in nx.average_neighbor_degree(RW_sample).items()], reverse=True)
# plt.loglog(degree_clustering, 'b-', marker='o')
# plt.title('Average Neighbor-Degree Rank of Sample Graph')
# plt.ylabel('Average Neighbor-Degree')
# plt.xlabel('rank')

plt.show()
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# degree = nx.clustering(RW_sample)
# #degree = nx.degree_histogram(RW_sample)          #返回图中所有节点的度分布序列
# x = range(len(degree))                             #生成x轴序列，从1到最大度
# y = [z / float(sum(degree)) for z in degree]
# #将频次转换为频率，这用到Python的一个小技巧：列表内涵，Python的确很方便：）
# plt.loglog(x, y, color="blue", linewidth=2)           #在双对数坐标轴上绘制度分布曲线
# plt.show()                                                          #显示图表
#--------------------------------------------------------------------------------------------

# render in two kinds of color
# spring_pos = nx.spring_layout(G)
# colors = []
# for node in G.nodes():
#     if node in RW_sample.nodes():
#         colors.append('b')
#     else:
#         colors.append('r')
# plt.figure()
# pos = nx.layout.spring_layout(G)
# nx.draw(G, spring_pos, node_color=colors, with_labels=True)
# plt.show()


# convert to node list
nodes = []
for node in RJ_sample.nodes():
    nodes.append([node, 2])
print(nodes)

class_nodes = []
for node in G.nodes():
    if node in RJ_sample.nodes():
        class_nodes.append([node, 2])
    else:
        class_nodes.append([node, 1])

# convert to edge list
edges = []
for edge in RJ_sample.edges():
    edges.append(edge)

orig_edges = []
for edge in G.edges():
    edges.append(edge)

# test csv
title = ['ID', 'Class']
test = pd.DataFrame(columns=title, data=class_nodes)
test.to_csv(classfile_path)


# # save as csv
# fw = open(nodefile_path, 'wb')
# writer = csv.writer(fw)
# writer.writerow(['ID', 'Class'])
# for row in nodes:
#     writer.writerow(row)
#
# fw = open(edgefile_path, 'wb')
# writer = csv.writer(fw)
# writer.writerow(['Source', 'Target'])
# for row in edges:
#     writer.writerow(row)
#
# fw = open(classfile_path, 'wb')
# writer = csv.writer(fw)
# writer.writerow(['ID', 'Class'])
# for row in nodes:
#     writer.writerow(row)
#
# fw = open(orig_edgefile_path, 'wb')
# writer = csv.writer(fw)
# writer.writerow(['Source', 'Target'])
# for row in edges:
#     writer.writerow(row)
#
#
# # get info
print('---------------------------------------------------------')
#print(nx.degree(G), nx.degree(RW_sample))
#print(nx.degree_histogram(G))
# print(nx.degree_centrality(G), nx.degree_centrality(RW_sample))
#nx.in_degree_centrality,nx.out_degree_centrality
# print(nx.clustering(G),  nx.clustering(RW_sample))
# print(nx.average_clustering(G), nx.average_clustering(RW_sample))
# print(nx.degree_assortativity_coefficient(G), nx.degree_assortativity_coefficient(RW_sample))
# print(nx.diameter(G), nx.diameter(RW_sample))
# print(nx.average_shortest_path_length(G), nx.average_shortest_path_length(RW_sample))
# print(nx.average_node_connectivity(G), nx.average_node_connectivity(RW_sample))
# print(nx.average_degree_connectivity(G), nx.average_degree_connectivity(RW_sample))  #均值连接度(均值近邻度)
# print(nx.average_neighbor_degree(G), nx.average_neighbor_degree(RW_sample))
# print(nx.closeness_centrality(G))
# print(nx.betweenness_centrality(G))
print('---------------------------------------------------------')


