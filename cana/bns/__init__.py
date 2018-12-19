# -*- coding: utf-8 -*-
"""
BNS Attractors Interface
==========================

This module interfaces CANA with the [B]oolean [N]etworks with [S]ynchronous update (BNS) :cite:`Dubrova:2011` to compute attractors.

	BNS is a software tool for computing attractors in Boolean Networks with Synchronous update.
	Synchronous Boolean networks are used for the modeling of genetic regulatory networks. 

	BNS implements the algorithm presented in which is based on a SAT-based bounded model checking.
	BNS uses much less space compared to BooleNet or other BDD-based approaches for computing attractors.
	It can handle several orders of magnitude larger networks. 



.. Note::

	You must have ``bns`` compiled for you system. Alternatively, you can download the binary from the `bns website <https://people.kth.se/~dubrova/bns.html>`_ directly.
	Last updated: December 2018.

"""
#	Copyright (C) 2018 by
#	Rion Brattig Correia <rionbr@gmail.com>
#	Alex Gates <ajgates@indiana.edu>
#	All rights reserved.
#	MIT license.
import os
import subprocess
import tempfile
import re
from cana.utils import strstates_to_numstate
# NetworkX decorators (see https://github.com/networkx/networkx/blob/master/networkx/utils/decorators.py)
from networkx.utils.decorators import open_file


_path = os.path.dirname(os.path.realpath(__file__))
""" Make sure we know what the current directory is """

def find_attractors(cnet, bnspath=_path, cleanup=True, l=None, u=None):
	"""Makes a subprocess call to ``bns`` supplying a temporary file with the boolean logic.

	Args:
		cnet (file,string) : A .cnet formated string or file.
		bnspath (string) : The path to the bns binary.
		cleanup (bool) : If cnet is a string, this function creates a temporary file.
			This forces the removal of this temp file.

		-l N (int) : restricts the search for atractors of lenght N and its factors.   
		      For example "bns -l 6" searches only for atractors of           
		      length 6 and its factors 3,2, and 1. Atractors of length 4,5 and  
		      any other of length larger than 6 will not be found.              
		      Limiting the search space might drastically reduce runtime and can
		      be an option to try when unrestricted search does not finish.     
		-u N (int) : set initial unfolding of transition relation to N levels instead  
		      of the default value which is equal to number of variables in     
		      Boolean network. This option does not impact the result but might 
		      increase or reduce total runtime. It is should not be used        
		      together with -l option.                      

	Example:
		
		```
		attractors = bns.find_attractors(BC.to_cnet(file=None, adjust_no_input=False))
		```

	Returns:
		list : the list of attractors
	"""
	
	# If is file, open the file
	if os.path.isfile(cnet):
		file = cnet
	
	# If string, Creates a Temporary File to be supplied to BNS
	elif isinstance(cnet, str):
		tmp = tempfile.NamedTemporaryFile(delete=cleanup)
		with open(tmp.name, 'w') as openfile:
			openfile.write(cnet)
		tmp.file.close()
		file = tmp.name
	else:
		raise InputError('The cnet input should be either a path to a .cnet file or a string containing the .cnet content')
	
	# Arguments are correct?
	if (l is not None) and (u is not None):
		raise TypeError('Arguments -l and -u should not be used together.')
	else:
		if l is not None:
			args = '-l {l:d}'.format(l=l)
		elif u is not None:
			args = 'u {u:d}'.format(u=u)
		else:
			args = ''

	attractors = {}
	cmd = "{path:s} {args:s} {file:s}".format(path=os.path.join(_path,'bns'), args=args, file=file )
	try:
		p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
		#
		i = 0
		attractor = set()
		for j,line in enumerate(p.stdout):
			cline = line.strip().replace('\n', '')
			# Very naive testing of lines. Could be improved upon.
			if 'Wrong format' in cline:
				raise TypeError("Wrong format of input file to BNS.")
			if 'Node' in cline and 'assumed to be constant' in cline:
				pass
			if 'Start searching for' in cline:
				pass
			if 'Total' in cline:
				pass
			if 'Depth' in cline:
				pass
			# 1 attractors of average length 1.00
			if 'of average length' in cline:
				pass
			#
			if 'Attractor' in cline:
				attractors[i] = attractor
			if _boolean_state_match(cline):
				attractor.add( strstates_to_numstate(cline) )
	except OSError:
		print("'BNS' could not be found! You must have it compiled or download the binary for your system from the 'bns' website (https://people.kth.se/~dubrova/bns.html).")

	return attractors

def _boolean_state_match(str, search=re.compile(r'[^0-1 ]').search):
	"""Verifies a string only contains '0' and '1's."""
	return not bool(search(str))