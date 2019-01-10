#!/usr/bin/env python

import sys, os
from commands import getoutput
from data_LIRS import *

blocktrain = './LIRS/blocktrain'
liblineartrain = './liblinear/train'
streamtrain = './streamsvm/streamtrain'
streampredict = './streamsvm/predict'

data_dir = ['/media/ssd/workload/SVM', '/media/3DXpoint/workload/SVM', '/media/disk/workload/SVM']
datasets = [web40, epsilon30, kdd40, higgs40]
solvers = ['BLOCK-L-D']

#measure time or acc
if len(sys.argv) != 2 or sys.argv[1] not in ['acc','time']:
	print('[Usage] ./go-compare-solvers-tkdd.py time|acc')
	exit(1)

if sys.argv[1] == 'time':
	acc = 0
elif sys.argv[1] == 'acc':
	acc = 1
	
#Check the existence of trainers
for trainer in [blocktrain, liblineartrain, streamtrain]:
	if not os.path.exists(trainer):
		os.system('cd LIRS;make')
		os.system('cd liblinear;make')
		os.system('cd streamsvm;make')

#Check the existence of log dirs
for dir in data_dir:
	if not os.path.exists('%s/log/time/vra' % dir) or not os.path.exists('%s/log/acc/vra' % dir):
		os.system('mkdir -p %s/log/time/vra' % dir)
		os.system('mkdir -p %s/log/acc/vra' % dir)

def execmd(cmd):
	#if not acc and os.path.exists('dropcache'):
	cmd = 'echo 3 > /proc/sys/vm/drop_caches; ' + cmd
	print '  $ %s' % (cmd)
	getoutput(cmd)

for dir in data_dir:
	for data in datasets:
		for s in solvers:
			if acc: 
				logname = '%s/log/acc/vra/%s#%s.acc' % (dir, data.name, s)
			else: 
				logname = '%s/log/time/vra/%s#%s' % (dir, data.name, s)
			print 'Generate "%s"' % logname
			if os.path.exists(logname):
				print 'Already ran this.. skip it'
				continue
			else:
				logtmp = logname + '.tmp'

			BLOCK_options = '%s -e 0.0001 -B -1 -c 1 -m 30 -p -r 1' % blocktrain

			if s == 'BLOCK-L-D':
				if not acc:
					if data.name == 'kdd40' or data.name == 'higgs40':
						execmd('cgexec -g memory:/mymemory %s -s 3 -M 30 %s/%s | tee %s' % \
								(BLOCK_options, dir, data.train, logtmp))
					else:
						execmd('cgexec -g memory:/mymemory %s -s 3 %s/%s | tee %s' % \
								(BLOCK_options, dir, data.train, logtmp))
				else :
					if data.name == 'kdd40' or data.name == 'higgs40':
						execmd('%s -s 3 -M 30 -t %s/%s.cbin %s/%s | tee %s' % \
								(BLOCK_options, dir, data.test, dir, data.train, logtmp))
					else:
						execmd('%s -s 3 -t %s/%s.cbin %s/%s | tee %s' % \
								(BLOCK_options, dir, data.test, dir, data.train, logtmp))

			# change log name to the correct one
			execmd('mv %s %s' % (logtmp, logname))


