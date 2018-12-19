from collections import Mapping, Set, Iterable
import numpy as np
import networkx as nx

class ThinDiGraph(nx.DiGraph):

	all_graph_dict = {'timestep':0}
	all_node_dict = {'state':'0'}
	all_edge_dict = {'weight':True}

	def single_graph_dict(self):
		return self.all_graph_dict

	def single_node_dict(self):
		return self.all_node_dict

	def single_edge_dict(self):
		return self.all_edge_dict

	#graph_attr_dict_factory = single_graph_dict
	#node_dict_factory = single_node_dict
	edge_attr_dict_factory = single_edge_dict


G = ThinDiGraph()
print G.graph
G.add_node(0, **{'label':'x1'})
G.add_node(1)
G.add_node(2, **{'label':'x3'})
G.add_edge(2,1, **{'weight':1})
G.add_edge(2,2)
print G._node
print G._adj
print '---'
print '> G.graph'
print G.graph
print '> G.nodes()'
for n,d in G.nodes(data=True):
	print n,d
for u,v,d in G.edges(data=True):
	print u,v,d
print G[2][1] is G[2][2]