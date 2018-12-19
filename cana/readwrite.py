# -*- coding: utf-8 -*-
"""
Read & Write
================

Read and write Boolean Networks from files in different formats.


"""
#	Copyright (C) 2018 by
#	Rion Brattig Correia <rionbr@gmail.com>
#	Alex Gates <ajgates@indiana.edu>
#	Thomas Parmer <tjparmer@indiana.edu>
#	All rights reserved.
#	MIT license.
from collections import defaultdict
from cana.boolean_network import BooleanNetwork
from cana.utils import *
import warnings
import re
# NetworkX decorators (see https://github.com/networkx/networkx/blob/master/networkx/utils/decorators.py)
from networkx.utils.decorators import open_file


#
# Read functions
#
@open_file(0, 'r')
def read_cnet(path, name=''):
	"""
	Instanciates a Boolean Network from a string in cnet format.

	Args:
		path (file or string) : A cnet format representation of a Boolean Network.
		name (string) : The network name.

	Returns:
		BN (BooleanNetwork) : The Boolean network.

	Examples:
		String should be structured as follow
		```
		#.v = number of nodes
		.v 1
		#.l = node label
		.l 1 node-a
		.l 2 node-b
		#.n = (node number) (in-degree) (input node 1) … (input node k)
		.n 1 2 4 5
		01 1 # transition rule
		```

	See also:
		:func:`read_logical` :func:`read_dict`

	Note: see examples for more information.
	"""
	#network_file = StringIO(string)
	logic = defaultdict(dict)

	line = path.readline()
	while line != "":
		if line[0] != '#' and line != '\n':
			# .v <#-nodes>
			if '.v' in line:
				number_of_nodes = int(line.split()[1])
				for inode in range(number_of_nodes):
					logic[inode] = {'name':'','in':[],'out':[]}
			# .l <node-id> <node-name>
			elif '.l' in line:
				logic[int(line.split()[1])-1]['name'] = line.split()[2]
			# .n <node-id> <#-inputs> <input-node-id>
			elif '.n' in line:
				inode = int(line.split()[1]) - 1
				indegree = int(line.split()[2])
				for jnode in range(indegree):
					logic[inode]['in'].append(int(line.split()[3 + jnode])-1)

				logic[inode]['out'] = [0 for i in range(2**indegree) if indegree > 0]

				logic_line = path.readline().strip()

				if indegree <= 0:
					if logic_line == '':
						# this is a node with no input, but not constant
						logic[inode]['in'] = [inode]
						logic[inode]['state'] = False
						logic[inode]['constant'] = False
						logic[inode]['out'] = [0,1]
					else:
						# this is a constant node
						logic[inode]['in'] = []
						logic[inode]['state'] = logic_line
						logic[inode]['constant'] = True
						logic[inode]['out'] = logic_line
				else:
					# this is a node with inputs
					while logic_line != '\n' and logic_line != '' and len(logic_line)>1:
						for nlogicline in expand_logic_line(logic_line):
							logic[inode]['out'][strstates_to_numstate(nlogicline.split()[0])] = int(nlogicline.split()[1])
						logic_line = path.readline().strip()

			# .e = end of path
			elif '.e' in line:
				break
		line = path.readline()
	return BooleanNetwork(logic=logic, name=name)


@open_file(0, 'r')
def read_logical(path, name=''):
	"""
	Reads a Boolean Network from a Boolean logic update rules format.

	Args:
		path (file or string) : A cnet format representation of a Boolean Network.
		name (string) : The network name.

	Returns:
		BN (BooleanNetwork) : The Boolean network object.

	Examples:
		Logic update rules should be structured as follow
		```
		#BOOLEAN RULES
		node_name*=node_input_1 [logic operator] node_input_2 ...
		```

	See also:
		:func:`from_string` :func:`read_dict`
	"""
	logic = defaultdict(dict)
	# parse lines to gather node names
	line = path.readline()
	i = 0
	while line != "":
		if line[0] == '#':
			line = path.readline()
			continue
		logic[i] = {'name': line.split("*")[0].strip(), 'in':[], 'out':[]}
		line = path.readline()
		i += 1

	# Parse lines again to determine inputs and output sequence
	path.seek(0)
	line = path.readline()
	i = 0
	while line != "":
		if line[0] == '#':
			line = path.readline()
			continue
		eval_line = line.split("=")[1] #logical condition to evaluate
		# RE checks for non-alphanumeric character before/after node name (node names are included in other node names)
		# Additional characters added to eval_line to avoid start/end of string complications
		input_names = [logic[node]['name'] for node in logic if re.compile('\W'+logic[node]['name']+'\W').search('*'+eval_line+'*')]
		input_nums = [node for input in input_names for node in logic if input==logic[node]['name']]
		logic[i]['in'] = input_nums
		# Determine output transitions
		logic[i]['out'] = output_transitions(eval_line, input_names)
		line = path.readline()
		i += 1

	return BooleanNetwork(logic=logic, name=name)


#
# Write functions
#
@open_file(1, 'wb')
def write_cnet(BN, path=None):
	"""Outputs the network logic to ``.cnet`` format, which is similar to the Berkeley Logic Interchange Format (BLIF).
	This is the format used by BNS to compute attractors.

	Args:
		BN (BooleanNetwork) : The Boolean network object.
		path (file; optional) : A string of the file to write the output to. If not supplied, a string will be returned.

	Returns:
		(string) : The ``.cnet`` format string.

	Note:
		See `BNS <https://people.kth.se/~dubrova/bns.html>`_ for more information.
	"""
	number_of_nodes = BN.number_of_nodes()

	bns_string = '# Network="{:s}"\n\n'.format(BN.name)
	bns_string += '# Total number of nodes\n'
	bns_string += '.v {number_of_nodes:d}\n\n'.format(number_of_nodes=number_of_nodes)
	
	# Node Labels
	bns_string += "# Node labels\n"
	for i,node in BN._nodes.items():
		bns_string += '.l {i:d} {name:s}\n'.format(i=i+1,name=node.name)
	bns_string += '\n'

	# Node inputs/outputs
	for i,node in BN._nodes.items():
		bns_string += '# Node {i:d}, name="{name:s}" k={k:d} inputs="{inputs:s}"\n'.format(i=i+1, name=node.name, k=node.k, inputs=','.join(map(str, [j+1 for j in node.inputs])) )
		bns_string += '.n {i:d} {k:d} {inputs:s}\n'.format( i=i+1 , k=node.k, inputs=' '.join(map(str, [j+1 for j in node.inputs])) )
		#
		if node.is_constant():
			# Constant
			bns_string += '1\n'
		else:
			# Not Constant; Print Positive transitions
			for numstate,output in enumerate(node.outputs, start=0):
				if output == '1':
					bns_string += '{numstate:s} {output:s}\n'.format( numstate=numstate_to_strstates(numstate, width=node.k) , output=output )
		bns_string += "\n"
	
	if path is None:
		return bns_string
	else:
		path.write(bns_string)


