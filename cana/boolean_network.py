# -*- coding: utf-8 -*-
"""
Boolean Network
================



"""
#	Copyright (C) 2017 by
#	Rion Brattig Correia <rionbr@gmail.com>
#	Alex Gates <ajgates@indiana.edu>
#	All rights reserved.
#	MIT license.
try:
    import cStringIO.StringIO as StringIO
except ImportError:
    from io import StringIO
import numpy as np
import networkx as nx
import random
from itertools import combinations
from collections import defaultdict
from joblib import Parallel, delayed
from cana.boolean_node import BooleanNode
from cana.utils import *
import warnings



class ThinDiGraph(nx.DiGraph):
	"""
	A smaller memory footprint class used to extend Networkx's `nx.DiGraph` into the `StateTransitionGraph`.
	In the STG edges contain no attributes.
	"""
	all_edge_dict = {'weight':1}
	def single_edge_dict(self):
		return self.all_edge_dict
	edge_attr_dict_factory = single_edge_dict


class StateTransitionGraph(ThinDiGraph):
	"""
	
	"""
	def __init__(self, incoming_graph_data=None, **attr):
		# Call __init__ on parent class
		super(self.__class__, self).__init__(incoming_graph_data=None, **attr)
		#
		self.graph['configuration_width'] = attr.get('configuration_width', None)
		self.graph['number_of_attractors'] = None

	def number_of_attractors(self):
		"""The number of attractors in the STG."""
		return self.graph['number_of_attractors']
	
	@property
	def attractors(self):
		return nx.get_node_attributes(self, 'attractor')
	
	@attractors.setter
	def attractors(self, attractors):
		"""Sets nodes as attractors. Usefull when attractors are computed elsewhere.

		Args:
			attractors (dict-of-lists) : A list of states, keyed by attractor.

		See also:
			:mod:`cana.bns`
		"""
		for i,attractor in attractors.items():
			for numstate in attractor:
				if not self.has_node(numstate):

					self.add_node(numstate, **{'label':numstate_to_strstates(numstate),'attractor':i})
				else:
					self.node[numstate]['attractor'] = i
		self.graph['number_of_attractors'] = i+1

		nx.set_node_attributes(self, dict_attractor, 'attractor')

	def find_attractors(self):
		"""Find the attractors of the boolean network via the STG.
		A full State-Transition-Graph is required for this function.
		In practice, attractors as defined as strongly connected components.
		
		Returns:
			attractors (dict-of-lists) : The list of states, keyed by attractor.
		
		Note:
			This method is slow for a large STG. Instead, consider using the BNS method
			then assigning the attractor states using `set_attractors`.

		See also:
			:mod:`cana.bns`
		"""
		attractors = nx.attracting_components(self)
		for i,attractor in enumerate(attractors, start=0):
			dict_attractor = { j:i for j in attractor}
			nx.set_node_attributes(self, dict_attractor, 'attractor')
		self.graph['number_of_attractors'] = i+1
		return self

	def attractor_subgraph(self):
		"""Returns a subnetwork where state nodes pertain to an attractor.
		This is 
		
		"""
		AG = nx.subgraph(self, self.attractors.keys())
		AG.name = 'AG: ' + self.name
		
		# Add edges to AG if there is path in the original graph.
		for i,j in combinations(np.arange(number_of_attractors), 2):
			# Get any state in the attractor
			i_numstate = self._attractors[i].pop()
			j_numstate = self._attractors[j].pop()
			# Ther is a (shortest) path between states.
			if nx.has_path(CSTG, i_numstate, j_numstate):
				CAG.add_edge(i,j)
			if nx.has_path(CSTG, j_numstate, i_numstate):
				CAG.add_edge(j,i)
		return AG

	def attractor_graph(self):
		"""Returns the Attractor Graph (AG).

		Args:
			CSTG (networkx.DiGraph) : A Controlled State-Transition-Graph (CSTG)

		Returns:
			AG (networkx.DiGraph) : The Controlled Attractor Graph (CAG)

		See also:
			:func:`attractor_driver_nodes`, :func:`perturbed_state_transition_graph`.
		"""
		configuration_width = self.graph['configuration_width']
		number_of_attractors = self.graph['number_of_attractors']
		state_attractor = self.attractors
		if len(state_attractor) <= 0:
			raise TypeError("State Transition Graph contains no attractors. Find attractors first.")
		
		# Converts dict {state:attractor} into {attractor:[states]}
		attractor_states = defaultdict(list)
		for state,attractor in state_attractor.items():
			attractor_states[attractor].append(state)

		AG = ThinDiGraph(name='Attractor Graph: ' + self.name)
		# Nodes
		for a, list_numstate in attractor_states.items():
			AG.add_node(a, **{'label':'|'.join([numstate_to_strstates(numstate, width=configuration_width) for numstate in list_numstate])})
		# Edges
		for i,j in combinations(np.arange(number_of_attractors), 2):
			# Get any state in the attractor
			i_numstate = attractor_states[i].pop(0)
			j_numstate = attractor_states[j].pop(0)
			# Ther is a (shortest) path between states.
			if nx.has_path(self, i_numstate, j_numstate):
				AG.add_edge(i,j)
			if nx.has_path(self, j_numstate, i_numstate):
				AG.add_edge(j,i)
			
		return AG

class BooleanNetwork(object):
	"""


	"""
	def __init__(self, name='', logic=None, rules={}, verbose=False, *args, **kwargs):

		self.name = name 							# Name of the Network
		self.timestep = 0							# A clock that keeps steptimes
		self.verbose = verbose

		# Intanciate Boolean Nodes
		self._nodes = dict()
		for i,node_logic in logic.items():
			name = node_logic.get('name')
			k = len(node_logic['in'])
			inputs = [j for j in logic[i]['in']]
			state = node_logic.get('state')
			outputs = node_logic['out']
			constant = node_logic.get('constant')
			node = BooleanNode(id=i, name=name, k=k, inputs=inputs, state=state, outputs=outputs, constant=constant)
			self._nodes[i] = node

		self.rules = rules			# A dict containing the perturbation schedule
		#
		#self.number_of_nodes = len(self.nodes) 				# Number of nodes
		#self.number_of_states = 2**self.number_of_nodes 	# Number of possible states in the network 2^N
		#
		#self._update_trans_func() 					# Updates helper functions and other variables

	def __str__(self):
		if len(self._nodes) > 10 :
			node_names = "[" + ','.join([node.name for i,node in self._nodes.items()[:4]]) + '...' + ','.join([node.name for i,node in self._nodes.items()[-4:]]) + "]"
		else:
			node_names = "[" + ','.join([node.name for i,node in self._nodes.items()]) + "]"
		return "<BNetwork(name={:s}, n={:d}, nodes={:s})>".format( self.name, self.number_of_nodes(), node_names )

	def __iter__(self):
		"""Iterate over the nodes. User: 'for n in BN'.

		Returns:
			niter (iterator) : An iterator over all nodes in the network.
		"""
		return iter(self._nodes.values())
	#
	# Property Methods
	# _name
	@property
	def name(self):
		"""Return the name of the network.

		Returns:
			name (string) : the name."""
		return self._name

	@name.setter
	def name(self, name=None):
		"""Set a new network name.

		Args:
			name (string) : new name.
		"""
		if not isinstance(name, (str,unicode)):
			raise TypeError("Network name must be string or unicode.")
		self._name = name

	@property
	def config(self):
		"""Returns the current network configuration. In practice, the current state of each node.

		Returns:
			config (dict) : the network current configuration.
		"""
		return ''.join( [ node.state for i,node in self._nodes.items() ] )

	@config.setter
	def config(self, config=None):
		"""Set the network on a particular configuration. In practice, sets the current state of each node.

		Args:
			config (iterable) : a list or string where items are the new node states in the configuration.
		"""
		if (len(config) != self.number_of_nodes() ):
			raise TypeError("Configuration length ({}) and number of nodes ({}) do not match.".format(len(config), self.number_of_nodes()) )
		if isinstance(config,dict):
			config = [s for i,s in config.items() ]
		
		for (i,node), newstate in zip(self._nodes.items(), config):
			node.state = newstate

	@property
	def rules(self):
		"""Return the schedule of control rules, if any.

		Returns:
			rules (dict) : The schedule of control rules.
		"""
		return self._rules

	@rules.setter
	def rules(self, rules=dict):
		"""Set a schedule of network control rules.
		
		Args:
			rules (dict) : a schedule of control rules to be applied to every node

		Examples:
			```
			bn.rules({
				# hold node 0 in Off state indefinitely.
				0 : {
					'time': None,
					'state': False
				},
				# hold node 3 in On state for 2 time steps.
				3 : {
					'time': 2,
					'state': True
				}
				# flips node 4 state
				4 : {
					'time': 1,
					state: None
				}
			})
			```
		"""
		if not isinstance(rules, dict):
			raise TypeError("Control rules attribute must be a dictionary")
		else:
			for k,v in rules.items():
				if ( (not isinstance(k, int)) or (k > self.number_of_nodes) or (k<0) ):
					raise TypeError("Control rules keys must match node ids in the network.")
				if ( ('time' not in v) or ('state' not in v) ):
					raise TypeError("Control rules values must include 'time' and 'state' descriptors.")
		self._rules = rules

	@property	
	def bias(self):
		"""Network Bias. The sum of individual node biases, divided by the number of nodes.
		Practically, it asks each node for their own bias.

		.. math:
			TODO

		See Also:
			:func:`~cana.boolean_node.bias`
		"""
		return sum( [node.bias for i,node in self._nodes.items() ] ) / self.number_of_nodes()

	#
	# Methods
	#
	def reset_timesteps(self):
		"""Resets the network internal timestep clock to zero."""
		self.timestep = 0	

	# _nodes
	def nodes(self, bunch=None):
		"""Return the list of nodes

		Args:
			bunch (list, optional) : a list of node ids to return.
		
		Returns:
			nodes (list) : the nodes in the Boolean Network.
		"""
		if bunch is not None:
			if isinstance(bunch, (int)):
				bunch = [bunch]
			return [node for i,node in self._nodes.items() if i in bunch]
		else:
			return [node for i,node in self._nodes.items()]

	def control(self, rule={}):
		"""Control nodes according to control rules.

		Args:
			rule (dict): A control rule.
		"""
		for i in rule['nodes']:
			# Does time dictates we must act on this node?
			acton = False
			if rule['time'] is not None:
				if rule['time'] > self.timestep:
					acton = True
			else:
				acton = True
			
			# If so, we control the node
			if acton:
				node = self._nodes[i]
				if rule['state'] is not None:
					node.state = rule['state']
				else:
					node.flip_state()
				if self.verbose:
					print("Controlling node {:,d} ({:s}) setting it to {:s}".format(i, node.name, node.state))
		
		return self

	def steps(self, n=1):
		"""Steps the boolean network dynamics 'n' steps.
		Args:
			n (int) : the number of steps to perform. Default is `1`.
		"""
		for t in np.arange(n):
			self.step()
		return self

	def step(self):
		"""One step in boolean network dynamics."""
		config = self.config
		for i,node in self._nodes.items():
			input_states = liststates_to_binstates( [ config[j] for j in node.inputs ] )
			node.step(input_states)

		# advance the internal timestep clock
		self.timestep += 1
		return self

	def number_of_nodes(self):
		"""Return the number of nodes in the network."""
		return len(self._nodes)
	
	def number_of_states(self):
		"""Return the number of states in the BN, accounting for constant nodes."""
		return 2 ** sum([1 for i,n in self._nodes.items() if not n.is_constant()])


	#
	# Graphs Methods
	#
	def structural_graph(self, remove_constants=False):
		"""Calculates and returns the structural graph of the boolean network.

		Args:
			remove_constants (bool) : Remove constants from the graph. Defaults to ``False``.

		Returns:
			G (networkx.Digraph) : The boolean network structural graph.
		"""
		name = "Structural Graph: " + self.name
		SG = nx.DiGraph(name=name)

		# Add Nodes
		SG.add_nodes_from( (i, {'label':node.name}) for i,node in self._nodes.items() )
		# Add Edges
		for i,target in self._nodes.items():
			for j in self._nodes[i].inputs:
				SG.add_edge(i, j, **{'weight':1.})

		if remove_constants:
			SG.remove_nodes_from([i for i,node in self._nodes.items() if node.is_constant()])
		#
		return SG

	def effective_graph(self, mode='input', bound='upper', threshold=None):
		"""Computes and returns the effective graph of the network.
		In practive it asks each :class:`~cana.boolean_node.BooleanNode` for their :func:`~cana.boolean_node.BooleanNode.effective_connectivity`.

		Args:
			mode (string) : Per "input" or per "node". Defaults to "node".
			bound (string) : The bound to which compute input redundancy.
			threshold (float) : Only return edges above a certain effective connectivity threshold.
				This is usefull when computing graph measures at diffent levels.

		Returns:
			(networkx.DiGraph) : directed graph

		See Also:
			:func:`~cana.boolean_node.BooleanNode.effective_connectivity`
		"""
		name = "Effective Graph: " + self.name + "(Threshold: {})".format(threshold)
		EG = nx.DiGraph(name=name)

		# Add Nodes
		EG.add_nodes_from( (i, {'label':node.name}) for i,node in self._nodes.items() )

		# Add Edges
		for i, node in self._nodes.items():

			if mode == 'node':
				raise Exception('TODO')

			elif mode == 'input':
				eff_conns = node.effective_connectivity(mode=mode, bound=bound, norm=False)
				for inputs,eff_conn in zip(node.inputs, eff_conns):
					# If there is a threshold, only add edges above the threshold. Else, return all edges.
					if ((threshold is None) and (eff_conn > 0)) or ((threshold is not None) and (eff_conn > threshold)):
						EG.add_edge(inputs, i, **{'weight':e_i})
			else:
				raise AttributeError('The mode you selected does not exist. Try "node" or "input".')

		return EG

	def activity_graph(self, threshold=None):
		"""
		Ghanbarnejad & Klemm (2012) EPL, 99
		"""
		name = "Activity Graph: " + self.name + "(Threshold: {})".format(threshold)
		AG = nx.DiGraph(name=name)

		# Add Nodes
		AG.add_nodes_from( (i, {'label':node.name}) for i,node in self._nodes.items() )

		# Add Edges
		for i, node in self._nodes.items():
			activities = node.activities()
			for inputs, activity in zip(node.inputs, activities):
				# If there is a threshold, only return those number above the threshold. Else, return all edges.
				if ((threshold is None) and (activity > 0)) or ((threshold is not None) and (activity > threshold)):
					AG.add_edge(inputs, i, **{'weight': activity})

		return AG

	def state_transition_graph(self):
		"""Creates and returns the full State Transition Graph (STG) for the Boolean Network.
		Note it accounts for constant nodes, thus the STG may be smaller than the full-STG.

		Returns:
			(networkx.DiGraph) : The state transition graph for the Boolean Network.
		"""
		config_bkp = self.config # Save configuration
		number_of_nodes = self.number_of_nodes()
		possible_number_of_states = 2**self.number_of_nodes()
		constant_nodes = {i:node.state for i,node in self._nodes.items() if node.is_constant()}
		#
		name = 'State Transition Graph: {:s}'.format(self.name)
		STG = StateTransitionGraph(name=name, configuration_width=number_of_nodes)

		# Nodes (states)
		"""
		if len(constant_nodes):
			# Only add states that are possible, given the constant variables and their states
			possible_states = set()
			for numstate in np.arange(possible_number_of_states):
				strstates = numstate_to_strstates(numstate, width=number_of_nodes)
				if not any([strstates[i] == state for i,state in constant_nodes.items()]):
					continue
				possible_states.add( numstate )

			STG.add_nodes_from( (numstate, {'label':numstate_to_strstates(numstate, width=number_of_nodes)}) for numstate in possible_states)
		else:
			# No constant nodes, add all possible states
			STG.add_nodes_from( (numstate, {'label':numstate_to_strstates(numstate, width=number_of_nodes)}) for numstate in np.arange(possible_number_of_states))
		"""
		# DEBUG
		STG.add_nodes_from( (numstate, {'label':numstate_to_strstates(numstate, width=number_of_nodes)}) for numstate in np.arange(possible_number_of_states))

		# Edges (transitions)
		for numstate in STG.nodes():
			self.config = numstate_to_strstates(numstate, width=number_of_nodes)
			STG.add_edge(numstate, strstates_to_numstate(self.step().config))
		#
		self.config = config_bkp # Reload configuration
		return STG

	def blind_trajectory(self, config, length=2):
		"""Computes a blind system trajectory of ``length`` steps without knowing the State Transition Graph (STG).
		Usefull when the STG is too large to be fully computed.
		
		Args:
			config (dict) : the network current configuration.
			length (int) : the length of steps to take.

		Returns:
			trajectory (list): a list of 
		"""
		if not self._is_valid_config(config):
			raise TypeError("Configuration '{:s}' is not valid considering constant nodes.".format(config))
		self.config = config
		trajectory = [config]
		for istep in range(length):
			trajectory.append(self.step().config)
		return trajectory

	def trajectory_to_attractor(self, config, *args, **kwargs):
		"""Computes the trajectory starting at ``initial`` until it reaches a attractor.

		Args:
			initial (string): the initial state.
		
		Returns:
			(list): the state trajectory between initial and the final attractor state.

		Note:
			This trajectory is garanteed since it requires previous knowledge of the attractors.
		"""
		if not self._is_valid_config(config):
			raise TypeError("Configuration '{:s}' is not valid considering constant nodes.".format(config))
		number_of_nodes = self.number_of_nodes()
		attractor_states = set( [numstate_to_strstates(state, width=number_of_nodes) for attset in self._attractors for state in attset] )

		self.config = config
		trajectory = [config]
		while (trajectory[-1] not in attractor_states):
			trajectory.append(self.step().config)

		return trajectory

	def _is_valid_config(self, config):
		"""Makes sure a specific configuration is valid under the node is_constant assumption."""
		constant_nodes = {i:node.state for i,node in self._nodes.items() if node.is_constant()}
		return False if not any([config[i] == state for i,state in constant_nodes.items()]) else True

	def node_as_constant(self, i=None, constant=False, state=0):
		"""Sets or unsets a Boolean variable (node) as a constant variable.

		Args:
			i (int) : The node id.
			constant (bool) : The boolean value for the node constant.
			state (int/string) : The state of the 

		Todo:
			This functions needs to better handle node_id and node_name
		"""
		if i not in self._nodes.keys():
			raise TypeError("Node not found in the network.")

		self._nodes[i].constant = constant
		self._nodes[i].state = state

	#
	# Dynamical Control Methods
	#
	def attractor_driver_nodes(self, STG, min_drivers=1, max_drivers=4, rule={'time':None,'state':None}, verbose=False):
		"""Get the minimum necessary driver nodes by iterating the combination of all possible driver nodes of length :math:`min <= x <= max`.

		Args:
			STG (StateTransitionGraph) : The State-Transition-Graph (STG)
			min_drivers (int) : Mininum number of driver nodes to search.
			max_drivers (int) : Maximum number of driver nodes to search.
			rule (dict) : `time` and `state` rules to be applied to the every driver set.

		Returns:
			(list) : The list of driver nodes found in the search.

		Note:
			This is an inefficient bruit force search, maybe we can think of better ways to do this?

		TODO:
			Parallelize the search on each combination. Each CSTG is independent and can be searched in parallel.

		See also:
			:func:`controlled_state_transition_graph`, :func:`controlled_attractor_graph`.
		"""
		nodeids = [i for i,node in self._nodes.items() if not node.is_constant()]

		attractor_drivers_found = []
		nr_drivers = min_drivers
		while (len(attractor_drivers_found) == 0) and (nr_drivers <= max_drivers):
			if verbose: print("Trying with {:d} Driver Nodes".format(nr_drivers))
			for drivers in combinations(nodeids, nr_drivers):
				drivers = list(drivers)
				tmprule = dict(rule)
				tmprule = [ {'nodes':[i],'time':tmprule['time'],'state':tmprule['state']} for i in drivers ]
				print tmprule
				CSTG = self.controlled_state_transition_graph(STG=STG, rules=tmprule)
				CAG = self.controlled_attractor_graph(CSTG)
				att_reachable_from = self.mean_reachable_attractors(CAG)

				if att_reachable_from == 1.0:
					attractor_drivers_found.append(dvs)
			# Add another driver node
			nr_dvs += 1

		if len(attractor_drivers_found) == 0:
			warnings.warn("No attractor control driver sets found. All subsets of size {:,d} to {:,d} searched.".format(min_drivers, max_drivers))

		return attractor_drivers_found

	def controlled_state_transition_graph(self, STG, rules):
		"""Returns the Controlled State-Transition-Graph (C-STG).
		In practice, it copies the original STG, applies control rules to nodes, and updates the C-STG.

		Args:
			STG (StateTransitionGraph) : The STG.
			rules (dict) : The schedule of control rules.

		Returns:
			(networkx.DiGraph) : The Controlled State-Transition-Graph.

		Note:
			Controls with timed rules are more computationaly expensive.
		
		See also:
			:func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
		"""
		for rule in rules:
			for i in rule['nodes']:
				if self._nodes[i].is_constant():
					warnings.warn("Cannot control a constant variable '%s'! Skipping" % self._nodes[i].name )

		number_of_nodes = self.number_of_nodes()
		attractors = STG.attractors
		if len(attractors) <= 0:
			raise TypeError("State Transition Graph contains no attractors. Find attractors first.")
		CSTG = STG.copy()
		CSTG.name = 'C-' + CSTG.name + '(' + str(rules) + ')'

		# The maximum number of steps in the pertubation rules
		max_steps = max( [rule['time'] for rule in rules] + [1] )

		for numstate in STG.nodes():
			strstates = numstate_to_strstates(numstate, width=STG.graph['configuration_width'])

			self.reset_timesteps()
			self.config = strstates

			# Some control rules need time to be taken into consideration
			for step in np.arange(max_steps):

				# Apply each control rule and revert Back to make sure all transitions were recorded.
				for rule in rules:
					controlling_nodes = rule['nodes']
					self.control(rule)
					control_numstate = self.config
					control_strstates = strstates_to_numstate(control_numstate)

					if not CSTG.has_edge(numstate,control_strstates):
						CSTG.add_edge(numstate, control_strstates, **{'jump':True,'cause':controlling_nodes})

					self.step()
					new_strstates = self.config
					new_numstate = strstates_to_numstate(new_strstates)
					if not CSTG.has_edge(numstate,new_numstate):
						CSTG.add_edge(numstate, new_numstate, **{'dynamic':True,'cause':controlling_nodes})

					# Revert to apply another Rule
					self.config = strstates

		return CSTG

	#
	# Reachability Measures
	#
	def mean_reachable_configurations(self, STG):
		"""Returns the Mean Fraction of Reachable Configurations

		Args:
			STG (networkx.DiGraph) : A (Controlled) State-Transition-Graph.
		Returns:
			(float) : Mean Fraction of Reachable Configurations
		"""
		if 'reachability' not in STG.graph:
			STG.graph['reachability'] = graph_reachability(STG)
		reachable_config = STG.graph['reachability']
		norm = ( 2.0**self.number_of_nodes() - 1 ) * self.number_of_states()
		return sum(reachable_config) / (norm)

	def mean_reachable_attractors(self, CAG, norm=True):
		"""The Mean Fraction of Reachable Attractors to a specific Controlled Attractor Graph (CAG).

		Args:
			CAG (networkx.DiGraph) : A Controlled Attractor Graph (CAG).

		Returns:
			(float) Mean Fraction of Reachable Attractors
		"""
		att_norm = (float( CAG.number_of_nodes() ) - 1.0) * len(CAG)

		if att_norm == 0:
			# if there is only one attractor everything is reachable
			return 1
		else:
			# otherwise find the reachable from each attractor
			if 'reachability' not in CAG.graph:
				CAG.graph['reachability'] = graph_reachability(CAG)
			attractor_reachable_config = CAG.graph['reachability']

			return sum(attractor_reachable_config) / (att_norm)

	#
	# Configuration Control Measures
	#
	def mean_controlable_configurations(self, STG, CSTG):
		"""The Mean Fraction of Controlable Configurations

		Args:
			STG (networkx.DiGraph) : The State-Transition-Graph.
			CSTG (networkx.DiGraph) : The Controlled State-Transition-Graph.

		Returns:
			(float) : Mean Fraction of Controlable Configurations.
		"""
		if 'reachability' not in STG.graph:
			STG.graph['reachability'] = graph_reachability(STG)
		reachable_from = STG.graph['reachability']

		if 'reachability' not in CSTG.graph:
			CSTG.graph['reachability'] = graph_reachability(CSTG)
		control_reachable_from = CSTG.graph['reachability']

		control_configs = control_reachable_from - reachable_from
		norm = ( 2.0**self.number_of_nodes() - 1 ) * self.number_of_states()

		return sum(control_configs) / (norm)


	# TO BE DEPRECATED
	def pinning_controlled_state_transition_graph(self, driver_nodes=[]):
		"""Returns a dictionary of Controlled State-Transition-Graph (CSTG) under the assumptions of
		pinning controllability:
		In practice, it copies the original STG, flips driver nodes (variables), and updates the CSTG.

		Args:
			driver_nodes (list) : The list of driver nodes.
		Returns:
			(networkx.DiGraph) : The Pinning Controlled State-Transition-Graph.
		See also:
			:func: `controlled_state_transition_graph`, :func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
		"""

		if self.keep_constants:
			for dv in driver_nodes:
				if dv in self.constants:
					warnings.warn("Cannot control a constant variable '%s'! Skipping" % self.nodes[dv].name )

		uncontrolled_system_size = self.number_of_nodes - len(driver_nodes)

		pcstg_dict = {}
		for att in self._attractors:
			dn_attractor_transitions = [tuple(''.join([self.num2bin(s)[dn] for dn in driver_nodes]) for s in att_edge)
			for att_edge in self._stg.subgraph(att).edges()]

			pcstg_states = [self.bin2num(binstate_pinned_to_binstate(
				statenum_to_binstates(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes) )
			for statenum in range(2**uncontrolled_system_size) for attsource, attsink in dn_attractor_transitions]

			pcstg = nx.DiGraph(name='STG: '+self.name)
			pcstg.name = 'PC-' + pcstg.name +' (' + ','.join(map(str,[self.nodes[dv].name for dv in driver_nodes])) + ')'

			pcstg.add_nodes_from( (ps, {'label':ps}) for ps in pcstg_states)

			for attsource, attsink in dn_attractor_transitions:
				for statenum in range(2**uncontrolled_system_size):
					initial = binstate_pinned_to_binstate(statenum_to_binstates(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes)
					pcstg.add_edge(self.bin2num(initial), self.bin2num(self.pinned_step(initial, pinned_binstate=attsink, pinned_var=driver_nodes)))

			pcstg_dict[tuple(att)] = pcstg

		return pcstg_dict

	# TO BE DEPRECATED
	def pinned_step(self, initial, pinned_binstate, pinned_var):
		"""Steps the boolean network 1 step from the given initial input condition when the driver variables are pinned
		to their controlled states.
		Args:
			initial (string) : the initial state.
			n (int) : the number of steps.
		Returns:
			(string) : The stepped binary state.
		"""
		# for every node:
		#   node input = breaks down initial by node input
		#   asks node to step with the input
		#   append output to list
		# joins the results from each node output
		assert len(initial) == self.number_of_nodes
		return ''.join( [ str(node.step( ''.join(initial[j] for j in self.logic[i]['in']) ) ) if not (i in pinned_var) else initial[i] for i,node in enumerate(self.nodes, start=0) ] )

	# TO BE DEPRECATED
	def pinning_controlled_state_transition_graph(self, driver_nodes=[]):
		"""Returns a dictionary of Controlled State-Transition-Graph (CSTG) under the assumptions of
		pinning controllability.

		In practice, it copies the original STG, flips driver nodes (variables), and updates the CSTG.

		Args:
			driver_nodes (list) : The list of driver nodes.

		Returns:
			(networkx.DiGraph) : The Pinning Controlled State-Transition-Graph.

		See also:
			:func: `controlled_state_transition_graph`, :func:`attractor_driver_nodes`, :func:`controlled_attractor_graph`.
		"""
		if self.keep_constants:
			for dv in driver_nodes:
				if dv in self.constants:
					warnings.warn("Cannot control a constant variable '%s'! Skipping" % self.nodes[dv].name )
		uncontrolled_system_size = self.number_of_nodes - len(driver_nodes)
		pcstg_dict = {}
		for att in self._attractors:
			dn_attractor_transitions = [tuple(''.join([self.num2bin(s)[dn] for dn in driver_nodes]) for s in att_edge)
			for att_edge in self._stg.subgraph(att).edges()]
			pcstg_states = [self.bin2num(binstate_pinned_to_binstate(
				statenum_to_binstates(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes) )
			for statenum in range(2**uncontrolled_system_size) for attsource, attsink in dn_attractor_transitions]
			pcstg = nx.DiGraph(name='STG: '+self.name)
			pcstg.name = 'PC-' + pcstg.name +' (' + ','.join(map(str,[self.nodes[dv].name for dv in driver_nodes])) + ')'
			pcstg.add_nodes_from( (ps, {'label':ps}) for ps in pcstg_states)
			for attsource, attsink in dn_attractor_transitions:
				for statenum in range(2**uncontrolled_system_size):
					initial = binstate_pinned_to_binstate(statenum_to_binstates(statenum, base=uncontrolled_system_size), attsource, pinned_var=driver_nodes)
					pcstg.add_edge(self.bin2num(initial), self.bin2num(self.pinned_step(initial, pinned_binstate=attsink, pinned_var=driver_nodes)))
			pcstg_dict[tuple(att)] = pcstg
		return pcstg_dict

	# TO BE DEPRECATED
	def fraction_pinned_attractors(self, pcstg_dict):
		"""Returns the Number of Accessible Attractors
		Args:
			pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

		Returns:
			(int) : Number of Accessible Attractors
		"""
		reached_attractors = []
		for att, pcstg in pcstg_dict.items():
			pinned_att = list(nx.attracting_components(pcstg))
			print(set(att), pinned_att)
			reached_attractors.append(set(att) in pinned_att)
		return sum(reached_attractors) / float(len(pcstg_dict))

	# TO BE DEPRECATED
	def fraction_pinned_configurations(self, pcstg_dict):
		"""Returns the Fraction of successfully Pinned Configurations

		Args:
			pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

		Returns:
			(list) : the Fraction of successfully Pinned Configurations to each attractor
		"""
		pinned_configurations = []
		for att, pcstg in pcstg_dict.items():
			att_reached = False
			for wcc in nx.weakly_connected_components(pcstg):
				if set(att) in list(nx.attracting_components(pcstg.subgraph(wcc))):
					pinned_configurations.append(len(wcc)/ len(pcstg))
					att_reached = True
			if not att_reached:
				pinned_configurations.append(0)

		return pinned_configurations

	def mean_fraction_pinned_configurations(self, pcstg_dict):
		"""Returns the mean Fraction of successfully Pinned Configurations

		Args:
			pcstg_dict (dict of networkx.DiGraph) : The dictionary of Pinned Controlled State-Transition-Graphs.

		Returns:
			(int) : the mean Fraction of successfully Pinned Configurations
		"""
		return sum(self.fraction_pinned_configurations(pcstg_dict)) / len(pcstg_dict)

	#
	# Dynamical Impact
	#
	def dynamical_impact(self, t=100, n_samples=0):
		"""Given an initial condition and the same configuration with node i perturbed, the system is run for t timesteps.
		The dynamical impact is the fraction of such configuration pairs that result in different configurations after t timesteps.

	#
		Args:
			t (int) : the number of time steps the system is run before impact is calculated.

			n_samples (int) : the number of samples used to approximate the dynamical impact of a node.
				if 0 then the full STG is used to calculate the true value instead of the approximation method.

		Returns:
			(vector) : An N-dimensional vector of dynamical impact for each node.
		"""
		impact_vec = [0.0 for inode in range(self.number_of_nodes)]
		for inode in range(self.number_of_nodes):
			if n_samples == 0:
				# use STG
				for statenum in range(self.number_of_states):
					config = self.num2bin(statenum)
					perturbed_config = flip_binstate_bit(config, inode)
					impact_vec[inode] += float(self.trajectory(config, length=t)[-1] != self.trajectory(perturbed_config, length=t)[-1]) / self.number_of_states
			else:
				# we sample configurations
				for isample in range(n_samples):
					rnd_config = "".join([random.choice(['0', '1']) for b in range(self.number_of_nodes)])
					perturbed_config = flip_binstate_bit(rnd_config, inode)
					impact_vec[inode] += float(self.trajectory(rnd_config, length=t)[-1] != self.trajectory(perturbed_config, length=t)[-1]) / n_samples
		return impact_vec

	def temporal_partial_derivative(self, t=100, n_samples=0, mode='tensor'):
		"""The partial derivative of node i on node j after t steps

		Args:
			t (int) : the number of time steps the system is run before impact is calculated.

			n_samples (int) : the number of samples used to approximate the dynamical impact of a node.
				if 0 then the full STG is used to calculate the true value instead of the approximation method.

			mode (str) : determines if we calculate the partial derivative over the whole trajectory
				'matrix' returns the partial derivative after t steps
				'tensor' : returns a tensor of the partial derivatives at each step of the t steps

		Returns:
			(vector) : the partial derivatives
		"""

		if mode == 'matrix':
			partial = np.zeros((self.number_of_nodes, self.number_of_nodes))
		elif mode=='tensor':
			partial = np.zeros((t, self.number_of_nodes, self.number_of_nodes))

		for inode in range(self.number_of_nodes):
			if n_samples == 0:
				# use STG
				config_genderator = (self.num2bin(statenum) for statenum in range(self.number_of_states))
				norm_term = self.number_of_states
			else:
				# sample configurations
				config_genderator = (random_binstate(self.number_of_nodes) for isample in range(n_samples))
				norm_term = n_samples


			for config in config_genderator:
				config_trajectory = self.trajectory(config, length=t)
				perturbed_config_trajectory = self.trajectory(flip_binstate_bit(config, inode), length=t)

				if mode == 'matrix':
					partial[inode] += [float(config_trajectory[-1][jnode] != perturbed_config_trajectory[-1][jnode]) for jnode in range(self.number_of_nodes)]

				elif mode=='tensor':
					for n_step in range(1,t+1):
						partial[n_step-1,inode] += [float(config_trajectory[n_step][jnode] != perturbed_config_trajectory[n_step][jnode]) for jnode in range(self.number_of_nodes)]
		return partial / norm_term


	#
	# Dynamics Canalization Map (DCM)
	#
	def dynamics_canalization_map(self, output=None, simplify=True, keep_constants=True):
		"""Computes the Dynamics Canalization Map (DCM).
		In practice, it asks each node to compute their Canalization Map and then puts them together, simplifying it if possible.

		Args:
			output (int) : The output DCM to return. Default is ``None``, retuning both [0,1].
			simplify (bool) : Attemps to simpify the DCM by removing thresholds nodes with :math:`\tao=1`.
			keep_constants (bool) : Keep or remove constants from the DCM.

		Returns:
			DCM (networkx.DiGraph) : a directed graph representation of the DCM.

		See Also:
			:func:`boolean_node.canalizing_map` for the CM and :func:`drawing.draw_dynamics_canalizing_map_graphviz` for plotting.
		"""
		CMs = []
		for i,node in self._nodes.items():
			if keep_constants or not node.constant:
				CMs.append( node.canalizing_map(output) )
		# https://networkx.readthedocs.io/en/stable/reference/algorithms.operators.html
		DCM = nx.compose_all(CMs)
		DCM.name = 'DCM: %s' % (self.name)

		if simplify:
			#Loop all threshold nodes
			threshold_nodes = [(n,d) for n,d in DCM.nodes(data=True) if d['type']=='threshold']
			for n,d in threshold_nodes:

				# Constant, remove threshold node
				if d['tau'] == 0:
					DCM.remove_node(n)

				# Tau == 1
				if d['tau'] == 1:
					in_nei = list(DCM.in_edges(n))[0]
					out_nei = list(DCM.out_edges(n))[0]

					neis = set( list(in_nei) + list(out_nei) )

					# Convert to self loop
					if (in_nei == out_nei[::-1]):
						DCM.remove_node(n)
						DCM.add_edge(in_nei[0],out_nei[1], **{'type':'simplified','mode':'selfloop'})
					# Link variables nodes directly
					elif not any([DCM.node[tn]['type']=='fusion' for tn in in_nei]):
						DCM.remove_node(n)
						DCM.add_edge(in_nei[0],out_nei[1], **{'type':'simplified','mode':'direct'})
		# Remove Isolates
		DCM.remove_nodes_from(nx.isolates(DCM))

		return DCM
	#
	# Get Node Names from Ids
	#
	def _get_node_name(self, id):
		"""Return the name of the node based on its id.

		Args:
			id (int): id of the node.

		Returns:
			name (string): name of the node.
		"""
		try:
			node = self._nodes[id]
		except:
			raise AttributeError("Node with id '%d' does not exist." % (id))
		else:
			return node.name

	def get_node_name(self, iterable=[]):
		"""Return node names. Optionally, it returns only the names of the ids requested.

		Args:
			iterable (int,list, optional) : The id (or list of ids) of nodes to which return their names.

		Returns:
			names (list) : The name of the nodes.
		"""
		# If only one id is passed, make it a list
		if not isinstance(iterable, list):
			iterable = [iterable]
		# No ids requested, return all the names
		if not len(iterable):
			return [n.name for n in self.nodes]
		# otherwise, use the recursive map to change ids to names
		else:
			return recursive_map(self._get_node_name, iterable)
	#
	# Plotting Methods
	#
	def derrida_curve(self, nsamples=10, random_seed=None, method='random'):
		"""The Derrida Curve (also reffered as criticality measure :math:`D_s`).
		When "mode" is set as "random" (default), it would use random sampling to estimate Derrida value
		If "mode" is set as "sensitivity", it would use c-sensitivity to calculate Derrida value (slower)
		You can refer to :cite:'kadelka2017influence' about why c-sensitivity can be used to caculate Derrida value

		Args:
			nsamples (int) : The number of samples per hammimg distance to get.
			random_seed (int) : The random state seed.
			method (string) : specify the method you want. either 'random' or 'sensitivity'

		Returns:
			(dx,dy) (tuple) : The dx and dy of the curve.
		"""
		random.seed(random_seed)
		number_of_nodes = self.number_of_nodes()

		dx = np.linspace(0, 1, number_of_nodes)
		dy = np.zeros(number_of_nodes)

		if method == 'random':
			# for each possible hamming distance between the starting states
			for hamm_dist in range(1, number_of_nodes + 1):

				# sample nsample times
				for isample in range(nsamples):
					rnd_config = [random.choice(['0', '1']) for b in range(number_of_nodes)]
					perturbed_var = random.sample(range(number_of_nodes), hamm_dist)
					perturbed_config = [ flip_bit(rnd_config[ivar]) if ivar in perturbed_var else rnd_config[ivar] for ivar in range(number_of_nodes) ]
					dy[hamm_dist-1] += hamming_distance(self.step(rnd_config), self.step(perturbed_config))

			dy /= float(number_of_nodes * nsamples)
		elif method == 'sensitivity':
			for hamm_dist in range(1, number_of_nodes + 1):
				dy[hamm_dist-1] = sum( [node.c_sensitivity(hamm_dist, mode='forceK', max_k=number_of_nodes) for i,node in self._nodes.items()] ) / number_of_nodes

		return dx, dy



