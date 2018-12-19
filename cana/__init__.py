__package__ = 'cana'
__title__ = u'CANAlization: Control & Redundancy in Boolean Networks'
__description__ = u'This package implements a series of methods used to study control, canalization and redundancy in Boolean Networks.'

__author__ = """\n""".join([
	'Rion Brattig Correia <rionbr@gmail.com>',
	'Alex Gates <ajgates@umail.iu.edu>',
	'Xuan Wang <xw47@indiana.edu>'
	'Thomas Parmer <tjparmer@indiana.edu>',
	'Etienne Nzabarushimana <enzabaru@indiana.edu>',
	'Luis M. Rocha <rocha@indiana.edu>'
])

__copyright__ = u'2018, Correia, R. B., Gates, A., Rocha, L. M.'

__version__ = '0.0.4'
__release__ = '0.0.4-alpha'
#
# Imports in order
#
import cana.utils

from cana.readwrite import *

from cana.boolean_node import *
from cana.boolean_network import *