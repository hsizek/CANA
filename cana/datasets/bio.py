# -*- coding: utf-8 -*-
"""
Biological Boolean Networks
=================================

Some of the commonly used biological boolean networks


"""
#   Copyright (C) 2017 by
#   Alex Gates <ajgates@gmail.com>
#   Rion Brattig Correia <rionbr@gmail.com>
#   Thomas Parmer <tjparmer@indiana.edu>
#   All rights reserved.
#   MIT license.
import os
import cana


_path = os.path.dirname(os.path.realpath(__file__))
""" Make sure we know what the current directory is """


def THALIANA():
	"""Boolean network model of the control of flower morphogenesis in Arabidopsis thaliana 

	The network is defined in :cite:`Chaos:2006`.

	Returns:
		(BooleanNetwork)
	"""
	return cana.read_cnet(_path + '/thaliana.txt', name="Arabidopsis Thaliana")

def DROSOPHILA(cells=1):
	"""Drosophila Melanogaster boolean model.
	This is a simplification of the original network defined in :cite:`Albert:2008`.
	In the original model, some nodes receive inputs from neighboring cells.
	In this single cell network, they are condensed (nhhnHH) and treated as constants.

	There is currently only one model available, where the original neighboring cell signals are treated as constants.
	
	Args:
		cells (int) : Which model to return. Default=1

	Returns:
		(BooleanNetwork)
	"""
	if cells == 1:
		return cana.read_cnet(_path + '/drosophila_single_cell.txt', name="Drosophila Melanogaster")
	else:
		raise AttributeException('Only single (1) cell drosophila boolean model currently available.')

def BUDDING_YEAST():
	"""The network is defined in :cite:`Fangting:2004`.
	
	Returns:
		(BooleanNetwork)
	"""
	return cana.read_cnet(_path + '/yeast_cell_cycle.txt', name="Budding Yeast Cell Cycle")

def MARQUESPITA():
	"""Boolean network used for the Two-Symbol schemata example.

	The network is defined in :cite:`Marques-Pita:2013`.

	Returns:
		(BooleanNetwork)
	"""
	return cana.read_cnet(_path + '/marques-pita_rocha.txt', name="Marques-Pita & Rocha")


def LEUKEMIA():
	"""Boolean network model of survival signaling in T-LGL leukemia  

	The network is defined in :cite:`Zhang:2008`.

	Returns:
		(BooleanNetwork)
	"""
	return cana.read_logical(_path + '/leukemia.txt', name="Leukemia")

def BREAST_CANCER():
	"""Boolean network model of signal transduction in ER+ breast cancer 

	The network is defined in :cite:`Zanudo:2017`.

	Returns:
		(BooleanNetwork)
	"""
	return cana.read_logical(_path + '/breast_cancer.txt', name="Breast Cancer")
