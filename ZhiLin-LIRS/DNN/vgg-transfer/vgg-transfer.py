import tensorflow as tf
import numpy as np
import time, random, sys, struct, threading
from queue import Queue
import tools
import VGG

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Define parameters
#DATA_DIR = "/home/xpoint/TensorFlow/ImageNet/imagenet_1280K/"   #Disk
DATA_DIR = "/media/ssd/ImageNet/imagenet_1280K/"   #SSD
MODEL_PATH = sys.argv[4]
NUM_INSTANCE = 1281152  #1281167
NUM_THREADS = 1
BATCH_SIZE = int(sys.argv[2])
MAX_EPOCH = 30
Q_MAXSIZE = int(sys.argv[3])
LEARNING_RATE = 0.01
VAL_NUM = 49984 #50000
VAL_BATCH_SIZE = 32
USE_SELU = 1

if sys.argv[1] == 'fix_order':
	RANDOM_SHUFFLE = 0
	TF_QUEUE = 0
if sys.argv[1] == 'random_shuffle':
	RANDOM_SHUFFLE = 1
	TF_QUEUE = 0
if sys.argv[1] == 'tf_queue':
	RANDOM_SHUFFLE = 0
	TF_QUEUE = 1

#Get one image [256*256*3, 1] from binary file	
def get_image(index, train_img_dir, train_label_dir):

	image_offset = 0 + index*(256*256*3)
	train_img_dir.seek(image_offset, 0)
	#img = struct.unpack('>196608B', train_img_dir.read(196608))
	#img = np.array(img).astype(np.float32)
	buf = train_img_dir.read(196608)
	img = np.fromstring(buf, dtype='>B').astype(np.float32)
	
	label_offset = 0 + index*(2)
	train_label_dir.seek(label_offset, 0)
	#label = struct.unpack('>1H', train_label_dir.read(2))
	#label = np.array(label).astype(np.float32)
	buf = train_label_dir.read(2)
	label = np.fromstring(buf, dtype='>H').astype(np.float32)
	
	img = (img/255)
	label = label-1
	
	return img, label	

#Enqueue the training data into Queue q
def enqueue(index_list, q, train_img_dir, train_label_dir):
	
	for index in index_list:
		image, label = get_image(index, train_img_dir, train_label_dir)
		q.put([image, label])

	train_img_dir.close()
	train_label_dir.close()	

#Dequeue and form a batch size of data (Used in random_shuffle mode)
def next_batch(q, batch_size = BATCH_SIZE):
	example_batch=[]
	label_batch=[]

	for i in range(batch_size):
		example, label = q.get()
		example_batch.append(example)
		label_batch.append(label)
		
	example_batch = np.array(example_batch).astype(np.float32)
	label_batch = np.array(label_batch).astype(np.int32)

	return example_batch, label_batch

#Dequeue and form a batch size of data (Used in TF_queue mode)
#It will random shuffle the q first.
def next_batch_TF(q, example_list, notTail, batch_size = BATCH_SIZE):
	example_batch=[]
	label_batch=[]
	
	while(len(example_list)<Q_MAXSIZE and notTail):
		example_list.append(q.get())
	
	#Random shuffle the queue then dequeue
	random.shuffle(example_list)

	for i in range(batch_size):
		example, label = example_list.pop()
		example_batch.append(example)
		label_batch.append(label)
		
	example_batch = np.array(example_batch).astype(np.float32)
	label_batch = np.array(label_batch).astype(np.int32)

	return example_batch, label_batch

#Create threads to do the enqueue process
def Create_threads(training_order, train_img_list, train_label_list):
	threads = []
	for i in range(NUM_THREADS):
		train_img_dir = open(train_img_list[i], 'rb')
		train_label_dir = open(train_label_list[i], 'rb')
		t = threading.Thread(target=enqueue, \
			args=(training_order[i],q,train_img_dir,train_label_dir))
		t.start()
		threads.append(t)
	return threads

#Define model
'''
with tf.name_scope('model'):
	
	x = tf.placeholder("float32", shape=[None, 256*256*3], name='x_in')
	y_raw = tf.placeholder("int32", shape=[None, 1])
	y_ = tf.cast(tf.one_hot(tf.reshape(y_raw, [-1]), depth=1000), tf.float32)
	learning_rate = tf.placeholder("float32")
	keep_prob = tf.placeholder('float32')
	
	x_image = tf.reshape(x, [-1,256,256,3])
	
	CONV_1 = _defineConv(x_image, 64, use_selu=USE_SELU)
	CONV_2 = _defineConv(CONV_1, 64, use_selu=USE_SELU)
	POOL_1 = _defineMaxPooling(CONV_2)
	
	CONV_3 = _defineConv(POOL_1, 128, use_selu=USE_SELU)
	CONV_4 = _defineConv(CONV_3, 128, use_selu=USE_SELU)
	POOL_2 = _defineMaxPooling(CONV_4)
	
	CONV_5 = _defineConv(POOL_2, 256, use_selu=USE_SELU)
	CONV_6 = _defineConv(CONV_5, 256, use_selu=USE_SELU)
	CONV_7 = _defineConv(CONV_6, 256, use_selu=USE_SELU)
	POOL_3 = _defineMaxPooling(CONV_7)
	
	CONV_8 = _defineConv(POOL_3, 512, use_selu=USE_SELU)
	CONV_9 = _defineConv(CONV_8, 512, use_selu=USE_SELU)
	CONV_10 = _defineConv(CONV_9, 512, use_selu=USE_SELU)
	POOL_4 = _defineMaxPooling(CONV_10)
	
	CONV_11 = _defineConv(POOL_4, 512, use_selu=USE_SELU)
	CONV_12 = _defineConv(CONV_11, 512, use_selu=USE_SELU)
	CONV_13 = _defineConv(CONV_12, 512, use_selu=USE_SELU)
	POOL_5 = _defineMaxPooling(CONV_13, name='POOL_5')
	
	flat = tf.layers.flatten(POOL_5)
	
	if USE_SELU:
		FC_1 = _defineDense(flat, 4096, tf.nn.selu)
		DROP_1 = tf.nn.dropout(FC_1, keep_prob)
		FC_2 = _defineDense(DROP_1, 4096, tf.nn.selu)
		DROP_2 = tf.nn.dropout(FC_2, keep_prob)
		
	else:
		FC_1 = _defineDense(flat, 4096, tf.nn.relu)
		DROP_1 = tf.nn.dropout(FC_1, keep_prob)
		FC_2 = _defineDense(DROP_1, 4096, tf.nn.relu)
		DROP_2 = tf.nn.dropout(FC_2, keep_prob)
	y_conv = _defineDense(DROP_2, 1000, None)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)
	loss = tf.reduce_mean(cross_entropy)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
'''
	
if __name__ == '__main__':

	#Create a queue with size Q_MAXSIZE to buffer the training data
	q =Queue(maxsize = Q_MAXSIZE)
	
	#Create the training order list
	training_order = np.zeros((NUM_THREADS, NUM_INSTANCE)).astype(np.int32)
	for i in range(NUM_THREADS):
		for j in range(NUM_INSTANCE):
			training_order[i][j] = j

	
	#Create the training file list
	train_img_list = []
	train_label_list = []

	for i in range(1):
		train_img_list.append(DATA_DIR + "train-images")
		train_label_list.append(DATA_DIR + "train-labels")

		
	#Reader threads list
	threads = []
	
	
	#Example list used in TF_queue
	example_list = []
	
	#-----------------------------------------------------											 
	x = tf.placeholder("float32", shape=[None, 256*256*3], name='x_in')
	x_reshape = tf.reshape(x, [-1,256,256,3])
	y_raw = tf.placeholder("int32", shape=[None, 1])
	y_ = tf.cast(tf.one_hot(tf.reshape(y_raw, [-1]), depth=1000), tf.float32)
	#learning_rate = tf.placeholder("float32")
	#keep_prob = tf.placeholder('float32')

    
	logits = VGG.VGG16(x_reshape, 1000, True)
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_,name='cross-entropy')
	loss = tf.reduce_mean(cross_entropy, name='loss')
	correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(y_, 1))
	correct = tf.cast(correct, tf.float32)
	accuracy = tf.reduce_mean(correct)
    
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
	train_op = optimizer.minimize(loss)
    

	
	
	#-----------------------------------------------------
	
	#Initialize timing parameter
	loadtime = 0
	traintime = 0
	startTime = time.time() 
	
	
	#Initialize TensorFlow
	saver = tf.train.Saver()
	init_op = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init_op)
	
	#saver.restore(sess, "models/model.ckpt")
	# load the parameter file, assign the parameters, skip the specific layers
	pre_trained_weights = './vgg16.npy'
	data_dict = np.load(pre_trained_weights, encoding='latin1').item()
	for key in data_dict:
		if key not in ['fc6','fc7','fc8']:
			with tf.variable_scope(key, reuse=True):
				for subkey, data in zip(('weights', 'biases'), data_dict[key]):
					sess.run(tf.get_variable(subkey).assign(data))
	
	for k in range(NUM_THREADS):
		random.shuffle (training_order[k])
	
	step = 0
	#Start training
	for i in range(MAX_EPOCH):

		#If the mode is RANDOM_SHUFFLE, shuffle the training order list first
		if RANDOM_SHUFFLE:
			for k in range(NUM_THREADS):
				random.shuffle (training_order[k])
		
		#Create threads(Readers) to read the training data 
		threads = Create_threads(training_order, train_img_list, train_label_list)
		
		
		#Inner iterations in range(Total data/batch size)
		for j in range(int(NUM_INSTANCE*NUM_THREADS/BATCH_SIZE)):
			
			step += 1
		
			#Start getting batch from Queue q
			start_getbatch = time.time()
			
			#import data for TensorFlow input pipeline
			if TF_QUEUE:
				if(j<int(NUM_INSTANCE*NUM_THREADS/BATCH_SIZE)-(Q_MAXSIZE/BATCH_SIZE-1)):
					notTail = 1
				else:
					notTail = 0
				batch = next_batch_TF(q, example_list, notTail, BATCH_SIZE)
			
			#import data for fix_order and random shuffle
			if not TF_QUEUE:
				batch = next_batch(q, BATCH_SIZE)

			loadtime += (time.time()-start_getbatch)
			
			
			#Start training process
			start_train = time.time()
			_, train_accuracy, train_loss=sess.run([train_op, accuracy, loss],feed_dict={x: batch[0], y_raw: batch[1]})
			
			#_, train_accuracy, train_loss, v_POOL_5=sess.run([train_step, accuracy, loss, POOL_1],feed_dict={x: batch[0], y_raw: batch[1]})
			traintime += (time.time()-start_train)
			
			print ("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g"%(i, j, train_loss, train_accuracy))

			
			if j%4000 == 0 and (i!=0 or j!=0):
				#Create threads and enqueue validation data
				v_q =Queue(maxsize = Q_MAXSIZE)
				
				v_threads = []
				v_image = open(DATA_DIR+'val-images', 'rb')
				v_label = open(DATA_DIR+'val-labels', 'rb')
				v_order = np.zeros(VAL_NUM).astype(np.int32)
				for k in range(VAL_NUM):
					v_order[k] = k
				
				t = threading.Thread(target=enqueue, args=(v_order,v_q,v_image,v_label))
				t.start()
				v_threads.append(t)
				
				#Evaluate the validation data
				val_accuarcy = 0
				val_loss = 0
				for k in range (int(VAL_NUM/VAL_BATCH_SIZE)):
					v_batch = next_batch(v_q, batch_size = VAL_BATCH_SIZE)
					v_accuarcy, v_loss = sess.run([accuracy, loss],feed_dict={x: v_batch[0], y_raw: v_batch[1]})
					val_accuarcy = val_accuarcy + v_accuarcy
					val_loss = val_loss + v_loss
				
				print ("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g, val_loss, %6g, val_accuracy, %g"\
						%(i, j, train_loss, train_accuracy, val_loss/(VAL_NUM/VAL_BATCH_SIZE), val_accuarcy/(VAL_NUM/VAL_BATCH_SIZE)))
				
				for thread in v_threads:
					thread.join()
		
		
					
		#Sync the threads 
		for thread in threads:
			thread.join()
			
		print("Total time: %f Traintime: %f Loadtime: %f" % ((time.time() - startTime), traintime, loadtime))
		save_path = saver.save(sess, MODEL_PATH)
