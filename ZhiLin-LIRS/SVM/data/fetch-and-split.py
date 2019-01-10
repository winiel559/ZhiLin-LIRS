#!/usr/bin/env python

import sys, os

datasets = ['webspam', 'epsilon', 'leisure', 'heart_scale']
datasets = ['webspam','epsilon', 'kdd']
datasets = ['epsilon']
trainsize = {'webspam': 280000, 'epsilon': 400000, \
		'leisure':368443, 'heart_scale':216}
datadir = './'

os.system('cd ./utils; make')

def subset(data):
	src = os.path.join(datadir, data)
	train = src + '.train'
	test = src + '.test'
	if os.path.exists(train) and os.path.exists(test):
		print('\tExists.')
	else:
		cmd = './utils/subset.py %s %s %s %s' %\
				(src, trainsize[data], train, test)
		print cmd
		os.system(cmd)

for data in datasets:
	print('Generate train/test for %s...' % (data))
	# Check data existence
	if data == 'epsilon':
		if os.path.exists('epsilon_normalized') and  os.path.exists('epsilon_normalized.t'):
			print('\tExist')
			continue
	elif data == 'webspam':
		if os.path.exists('webspam.train') and os.path.exists('webspam.test'):
			print('\tExist')
			continue

	# Fetch data from web
	if data == 'webspam':
		datapath = os.path.join(datadir, 'webspam')
		bz2path = os.path.join(datadir, 'webspam.bz2')
		print('Get webspam from web...')
		if os.path.exists(datapath):
			print('\tExist')
		else:
			if not os.path.exists(bz2path):
				cmd = 'wget -O %s http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/webspam_wc_normalized_trigram.svm.bz2' % (bz2path)
				print cmd
				os.system(cmd)
			cmd = 'bunzip2 %s' % (bz2path)
			print cmd
			os.system(cmd)
	elif data == 'kdd':
		print('Get kddb from web...')
		for x in ['kddb','kddb.t']:
			datapath = os.path.join(datadir, x)
			bz2path = os.path.join(datadir, x+'.bz2')
			if os.path.exists(datapath):
				print('\tExist')
			else:
				if not os.path.exists(bz2path):
					cmd = 'wget -O %s http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/%s' % (bz2path, x+'.bz2')
					print cmd
					os.system(cmd)
				cmd = 'bunzip2 %s' % (bz2path)
				print cmd
				os.system(cmd)
				#modify labels to {-1,1}
				cmd = "sed -e 's/^0/-1/' %s > tmpfile" % datapath
				print cmd
				os.system(cmd)
				cmd = "mv tmpfile %s " % datapath
				print cmd
				os.system(cmd)                             
		continue # already have train and test
	elif data == 'epsilon':
		print('Get epsilon from web...')
		for x in ['epsilon_normalized','epsilon_normalized.t']:
			datapath = os.path.join(datadir, x)
			bz2path = os.path.join(datadir, x+'.bz2')
			if os.path.exists(datapath):
				print('\tExist')
			else:
				if not os.path.exists(bz2path):
					cmd = 'wget -O %s http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/%s' % (bz2path, x+'.bz2')
					print cmd
					os.system(cmd)
				cmd = 'bunzip2 %s' % (bz2path)
				print cmd
				os.system(cmd)
		continue # already have train and test
	elif data == 'leisure':
		print('leisure is not available online')
	elif data == 'heart_scale':
		cmd = 'wget -O heart_scale http://www.csie.ntu.edu.tw/~cjlin/libsvmtools\
				datasets/binary/heart_scale' 
		print cmd
		os.system(cmd)

	subset(data)

	if data == 'webspam':
		datapath = os.path.join(datadir,'webspam.train')
		rawdatapath = os.path.join(datadir,'webspam.train.raw')
		print('Generate %s' % rawdatapath)	
		if os.path.exists(rawdatapath):
			print('\tExist')
		else :
			cmd = "grep '^+1' %s > %s; grep '^-1' %s >> %s" %\
					(datapath, rawdatapath, datapath, rawdatapath)
			print cmd
			os.system(cmd)
