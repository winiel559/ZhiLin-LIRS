#!/usr/bin/env python
import sys, os
from data import *

datasets = [web40, epsilon30, leisure5, kdd40]
datasets = [higgs40]
datadir = '/media/ssd/workload/SVM'
blockspliter = '../BMF/blockspliter'

if not os.path.exists(blockspliter):
	os.system('cd ../BMF; make ')

def gensinglebin(srcpath, target):
	if os.path.exists(target):
		print ('\texists.')
		return
	if not os.path.exists(srcpath):
		print ('\tsource %s does not exist. Skip this data' % srcpath)
	else:
		singleblock = srcpath+'.1'
		cmd = '%s -S 1 -c %s %s' % (blockspliter, srcpath, singleblock)
		print cmd
		os.system(cmd)
		cmd = 'ln -fs %s/data/*1.bin %s' % (singleblock, target)
		print cmd
		os.system(cmd)

for data in datasets:
	trainpath = os.path.join(datadir, data.train)
	traincbin = os.path.join(datadir, '%s.cbin' % (data.train))
	testpath = os.path.join(datadir, data.test)
	testcbin = os.path.join(datadir, '%s.cbin' % (data.test))
	print('Generete single binary for %s ' % (data.name))
	gensinglebin(trainpath, traincbin)
	gensinglebin(testpath, testcbin)


