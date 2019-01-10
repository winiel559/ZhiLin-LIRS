#!/usr/bin/env python

import sys, os
from  commands import getoutput
from data import *

datasets = [leisure5, web40, epsilon30, web200, web400, web1000, webraw40, kdd40]
datasets = [epsilon30, web40, kdd40, higgs40]
#datasets = [heart_scale5]
datadir = '/media/ssd/workload/SVM/'
blockspliter = '../BMF/blockspliter'

if not os.path.exists(blockspliter):
	os.system('cd ../BMF; make ')

for data in datasets:
	target = os.path.join(datadir, '%s.%s' % (data.train, data.blocks))
	trainpath = os.path.join(datadir, data.train)
	print('Generete %s ...' % (data.name))
	if os.path.exists(target):
		print ('\texists.')
		continue
	if not os.path.exists(trainpath):
		print ('\tsource %s does not exist. Skip this data' % trainpath)
	else :
		cmd = '%s -S %s -c %s %s' % (blockspliter, data.blocks, trainpath, target)
		print cmd
		output = getoutput(cmd)
		time = output.split()[-1]
		open('data.py','a').write('%s.time = %s\n' % (data.name, time))
	


