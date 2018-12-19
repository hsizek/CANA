import cana
from cana import bns
from cana.datasets import bio, bools
import networkx as nx

#print('--- BNode ---')
#n = bools.COPYx1()
#print n
#print n.activities()
#asd

#N = cana.BooleanNode(name='OR', k=2, inputs=['i_1','i_2'], state='1', outputs=[0,1,1,1], constant='0', verbose=True)
#print(N)
#N.newstep(input="01")
#print(N)


print('--- BNetwork ---')
name = 'Gates&Rocha'
logic = {
	0 : {
		'name':'x1',
		'in':[],
		'out':'1',
		'constant':False,
		'state':True
	},
	1 : {
		'name':'x2',
		'in':[0],
		'out':'01',
		'constant':False,
		'state':True
	},
	2 : {
		'name':'x3',
		'in':[0,1],
		'out':'0111',
		'constant':False,
		'state':True
	},
}
#BN = cana.BooleanNetwork(name=name, logic=logic, verbose=True)
BN = bio.BUDDING_YEAST()
#BN = bio.THALIANA()

#BN.node_as_constant(i=10, constant=False, state=1)
#BN.node_as_constant(i=11, constant=False, state=1)
#print '--- I/O ---'
#cnetstr = BN.to_cnet()

print BN
for node in BN:
	print node
print "Number of nodes: {:,d}".format( BN.number_of_nodes() )
print "Number of states: {:,d}".format( BN.number_of_states() )

print cana.write_cnet(BN, 'test.txt')

print '--- STG ---'
stg = BN.state_transition_graph()

print "Number STG nodes: {:d}".format( stg.number_of_nodes() )
print "Number STG edges: {:d}".format( stg.number_of_edges() )

"""
for i,d in stg.nodes(data=True):
	print i,d
for u,v,d in stg.edges(data=True):
	print u,stg.node[u], v,stg.node[v], d
"""

print '-- Finding Attractors'
stg.find_attractors()
#file = cana.write_cnet(BN,'test_gates_rocha.txt')
#bns_attractors = bns.find_attractors(cana.write_cnet(BN))
#print bns_attractors

#stg.find_attractors()
stg_attractors = stg.attractors


print '> # Attractors:',stg.number_of_attractors()
print '> Attractors {numstate:id_attractor}:',stg.attractors

asd

print("--- (new) CSTG ---")
rules = {
	0 : {
		'time': None,
		'state': None
	},
	1 : {
		'time': None,
		'state': None
	}
}
rules = [
	{
		'nodes': [0],
		'time': None,
		'state': None
	},
	{
		'nodes': [2],
		'time': None,
		'state': None
	}
]
cstg = BN.controlled_state_transition_graph(stg, rules=rules)
print "#nodes:",cstg.number_of_nodes()
for i,d in cstg.nodes(data=True):
	print i,d
for u,v,d in cstg.edges(data=True):
	print u,cstg.node[u], v,cstg.node[v], d

print("--- CAG ---")
cag = cstg.attractor_graph()
print "#nodes:",cag.number_of_nodes()
for i,d in cag.nodes(data=True):
	print i,d
for u,v,d in cag.edges(data=True):
	print u,cag.node[u], '-', v,cag.node[v], d	

asd
print("--- Measures ---")

print BN.mean_reachable_configurations(cstg)
print BN.mean_controlable_configurations(stg,cstg)
print BN.mean_reachable_attractors(cag)

print("--- Structural Graph--")
sg = BN.structural_graph()
print sg
for n in sg.nodes(data=True):
	print n
for u,v,d in sg.edges(data=True):
	print u,sg.node[u], '-', v,sg.node[v], d	
"""
print '>> SET NODE 1 as CONSTANT'
BN.node_as_constant(2, True, state=1)

print BN
for node in BN:
	print node

print "Number of nodes: {:,d}".format( BN.number_of_nodes() )
print "Number of states: {:,d}".format( BN.number_of_states() )


stg = BN.state_transition_graph()
for i,d in stg.nodes(data=True):
	print i,d
for u,v,d in stg.edges(data=True):
	print stg.node[u], stg.node[v], d
"""


"""
BN.perturbations = {
	1 : {
		'time': None,
		'state': True
	},
	2 : {
		'time': None,
		'state': True
	}
}
"""
