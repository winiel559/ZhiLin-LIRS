#!/usr/bin/env python

import sys

class Data:
	def __init__(self, name, train, test, blocks, time, primal,bestC, display=""):
		self.name = name
		self.train = train
		self.test = test
		self.blocks = blocks
		self.time = time
		self.vwtime = 0
		self.primal = primal
		self.bestC = bestC
		if display: 
			self.display = display
		else:
			self.display = name

heart_scale5 = Data('heart_scale5', 'heart_scale', 'heart_scale', 5, 1, 120,1)
heart_scale10 = Data('heart_scale10', 'heart_scale', 'heart_scale', 10, 2, 120,1)
heart_scale20 = Data('heart_scale20', 'heart_scale', 'heart_scale', 20, 3, 120,1)

leisure5 = Data('leisure5', 'leisure.scale.train', 'leisure.scale.test', 5, 227.56,117478.790608,4,'yahoo-korea')
epsilon30 = Data('epsilon30', 'epsilon_normalized', 'epsilon_normalized.t', 30, 1566.83, 109668.255408,1,'epsilon')
web40 = Data('web40','webspam.train', 'webspam.test', 40, 1594.05,10277.335292,64,'webspam')
web200 = Data('web200','webspam.train', 'webspam.test', 200, 1662.66,10277.335292,64)
web400 = Data('web400','webspam.train', 'webspam.test', 400, 1905.3,10277.335292,64)
web1000 = Data('web1000','webspam.train', 'webspam.test', 1000, 2192.73, 10277.335292,64)
webraw40 = Data('webraw40','webspam.train.raw', 'webspam.test', 40, 1577.37,10277.335292,64)
kdd40 = Data('kdd40', 'kddb', 'kddb.t', 40, 548.916, 1884906.183602, 0.1,'kddcup10')
higgs40 = Data('higgs40', 'higgs', 'higgs.t', 40, 548.916, 1884906.183602, 0.1,'higgs')
#--------below region is for upating block generating time-------
heart_scale5.vwtime = 0.04
epsilon30.time = 630
web40.time = 770
web200.time = 781
web400.time = 786
web1000.time = 787
webraw40.time = 790
kdd40.time = 326
web40.time = 567
kdd40.time = 552
