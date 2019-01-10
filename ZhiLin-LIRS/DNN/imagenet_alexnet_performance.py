import tensorflow as tf
import numpy as np
import time, random, sys, struct, threading
from queue import Queue

#Define parameters
#DATA_DIR = "/home/xpoint/TensorFlow/ImageNet/imagenet_1280K/"
#DATA_DIR = "/media/ssd/ImageNet/imagenet_1280K/"
DATA_DIR = "/media/xpoint/imagenet_100K/"
TOTAL_INSTANCE = 100000 #1281024
NUM_INSTANCE = 10240 #1281024 #1281152  #1281167
NUM_THREADS = 1
BATCH_SIZE = int(sys.argv[2])
MAX_EPOCH = 3
Q_MAXSIZE = int(sys.argv[3])
LEARNING_RATE = 1e-4
VAL_NUM = 50000
VAL_BATCH_SIZE = 200
MODEL_PATH = sys.argv[4]

if sys.argv[1] == 'fix_order':
	RANDOM_SHUFFLE = 0
	TF_QUEUE = 0
if sys.argv[1] == 'random_shuffle':
	RANDOM_SHUFFLE = 1
	TF_QUEUE = 0
if sys.argv[1] == 'tf_queue':
	RANDOM_SHUFFLE = 0
	TF_QUEUE = 1
	
#Initialize weight
def weight_variable(shape, stddev=0.001):
	initial = tf.truncated_normal(shape, mean=0, stddev=stddev)
	return tf.Variable(initial)
	
#Initialize bias
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)
	
#Define convolutional layer
def conv2d(x, W, stride_y, stride_x, padding='SAME'):
	return tf.nn.conv2d(x, W, strides=[1, stride_y, stride_x, 1], padding=padding)
	
#Define max pooling layer with stride parameter	
def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding='SAME'):
	return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding)

#Local response normalization from AlexNet
def lrn(x, radius, alpha, beta, bias=1.0):
	return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
	
#Add CONV layer or FC layer	
def add_layer(inputs, shape, layer_type, stddev=0.001, act_func=None, stride_y=1, stride_x=1, padding='SAME', norm=True):
	Weights = weight_variable(shape, stddev)

	if layer_type == 'CONV':
		out_size = shape[3]
		biases = bias_variable([out_size])
		Wx_plus_b = conv2d(inputs, Weights, stride_y, stride_x, padding) + biases
		
		if norm:
			fc_mean, fc_var = tf.nn.moments(Wx_plus_b,axes=[0,1,2],)
			scale = tf.Variable(tf.ones([out_size]))
			shift = tf.Variable(tf.zeros([out_size]))
			epsilon = 0.001
			Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)

	if layer_type == 'FC':
		out_size = shape[1]
		biases = bias_variable([out_size])
		Wx_plus_b = tf.matmul(inputs, Weights) + biases
		
		if norm:
			fc_mean, fc_var = tf.nn.moments(Wx_plus_b,axes=[0],)
			scale = tf.Variable(tf.ones([out_size]))
			shift = tf.Variable(tf.zeros([out_size]))
			epsilon = 0.001
			Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, fc_mean, fc_var, shift, scale, epsilon)


	if act_func is None:
		outputs = Wx_plus_b
	else:
		outputs = act_func(Wx_plus_b)
	return outputs

#Get one image [256*256*3, 1] from binary file	
def get_image(index, train_img_dir, train_label_dir):

	start = time.time()
	
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
	
	elaped_time = time.time()-start
	
	return img, label, elaped_time

#Enqueue the training data into Queue q
def enqueue(index_list, q, train_img_dir, train_label_dir, epoch):
	
	total_enqueue = 0
	
	for index in range(epoch*NUM_INSTANCE, (epoch+1)*NUM_INSTANCE, 1):
		#print(index_list[index])
		image, label, elaped_time = get_image(index_list[index], train_img_dir, train_label_dir)
		total_enqueue += elaped_time
		q.put([image, label])
	
	print("Enqueue time: %f" % total_enqueue)

	train_img_dir.close()
	train_label_dir.close()	

#Dequeue and form a batch size of data (Used in random_shuffle mode)
def next_batch(q, batch_size = BATCH_SIZE):

	#start = time.time()
	example_batch=[]
	label_batch=[]

	for i in range(batch_size):
		example, label = q.get()
		example_batch.append(example)
		label_batch.append(label)
	
	#print("get_batch: %f" % (time.time()-start))

	return example_batch, label_batch

#Dequeue and form a batch size of data (Used in TF_queue mode)
#It will random shuffle the q first.
def next_batch_TF(q, example_list, notTail, batch_size = BATCH_SIZE):
	example_batch=[]
	label_batch=[]
	
	#start_example_list = time.time()
	while(len(example_list)<Q_MAXSIZE and notTail):
		example_list.append(q.get())
	#print("start_example_list: %f" % (time.time()-start_example_list))
	
	#start_example_list = time.time()
	#Random shuffle the queue then dequeue
	random.shuffle(example_list)
	#print("start_shuffle_list: %f" % (time.time()-start_example_list))

	for i in range(batch_size):
		example, label = example_list.pop()
		example_batch.append(example)
		label_batch.append(label)

	return example_batch, label_batch

#Create threads to do the enqueue process
def Create_threads(training_order, train_img_list, train_label_list, epoch):
	threads = []
	for i in range(NUM_THREADS):
		train_img_dir = open(train_img_list[i], 'rb')
		train_label_dir = open(train_label_list[i], 'rb')
		t = threading.Thread(target=enqueue, \
			args=(training_order[i],q,train_img_dir,train_label_dir, epoch))
		t.start()
		threads.append(t)
	return threads

#Define model	
with tf.name_scope('model'):
	
	x = tf.placeholder('float32', shape=[None, 256*256*3])
	y_raw = tf.placeholder('int32', shape=[None, 1])
	y_ = tf.cast(tf.one_hot(tf.reshape(y_raw, [-1]), depth=1000), tf.float32)
	keep_prob = tf.placeholder("float")
	
	x_image = tf.reshape(x, [-1,256,256,3])
	
	conv1 = add_layer(x_image, [11,11,3,96], 'CONV', stddev=0.001, act_func=tf.nn.relu, stride_y=4, stride_x=4, padding='VALID')
	pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID')
	
	conv2 = add_layer(pool1, [5,5,96,256], 'CONV', stddev=0.001, act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
	pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID')
	
	conv3 = add_layer(pool2, [3,3,256,384], 'CONV', stddev=0.001, act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
	
	conv4 = add_layer(conv3, [3,3,384,384], 'CONV', stddev=0.001, act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
	
	conv5 = add_layer(conv4, [3,3,384,256], 'CONV', stddev=0.001, act_func=tf.nn.relu, stride_y=1, stride_x=1, padding='SAME')
	pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')
	

	flat = tf.reshape(pool5, [-1, 6*6*256])
	
	FC_1 = add_layer(flat, [6*6*256, 4096], 'FC', stddev=0.1, act_func=tf.nn.relu)
	DROP_1 = tf.nn.dropout(FC_1, keep_prob)
	FC_2 = add_layer(DROP_1, [4096, 4096], 'FC', stddev=0.1, act_func=tf.nn.relu)
	DROP_2 = tf.nn.dropout(FC_2, keep_prob)
	y_conv = add_layer(DROP_2, [4096, 1000], 'FC', stddev=0.1)

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
	loss = tf.reduce_mean(cross_entropy)
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	
if __name__ == '__main__':

	#Create a queue with size Q_MAXSIZE to buffer the training data
	q =Queue(maxsize = Q_MAXSIZE)
	
	
	#Create the training order list
	training_order = np.zeros((NUM_THREADS, TOTAL_INSTANCE)).astype(np.int32)
	for i in range(NUM_THREADS):
		for j in range(TOTAL_INSTANCE):
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
	
	
	#Initialize timing parameter
	loadtime = 0
	traintime = 0
	shuffle_create_threads_time = 0
	sync_threads_time = 0
	total_time = 0
	startTime = time.time() 
	
	
	#Initialize TensorFlow
	saver = tf.train.Saver()
	init_op = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init_op)
	
	if RANDOM_SHUFFLE:
		for k in range(NUM_THREADS):
			random.shuffle (training_order[k])
	
	#Start training
	for i in range(MAX_EPOCH):

		start_shuffle_create_threads = time.time()
		
		#If the mode is RANDOM_SHUFFLE, shuffle the training order list first
		if RANDOM_SHUFFLE:
			for k in range(NUM_THREADS):
				random.shuffle (training_order[k])
		
		#Create threads(Readers) to read the training data 
		threads = Create_threads(training_order, train_img_list, train_label_list, i)
		
		shuffle_create_threads_time += (time.time()-start_shuffle_create_threads)
		
		
		#Inner iterations in range(Total data/batch size)
		startTime_2 = time.time() 
		for j in range(int(NUM_INSTANCE*NUM_THREADS/BATCH_SIZE)):
		
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
			_, train_accuracy, train_loss, cross_entropy_p, pool5_p = \
				sess.run([train_step, accuracy, loss, cross_entropy, pool5],\
				feed_dict={x: batch[0], y_raw: batch[1], keep_prob: 0.5})
			#print (pool5_p)
			traintime += (time.time()-start_train)
			
			print ("Epoch, %3d, step, %6d, train_loss, %6g, training_accuracy, %g"%(i, j, train_loss, train_accuracy))
			#print ("Cross Entropy", cross_entropy_p)
			#Print the time every 100 iters
		total_time += time.time() - startTime_2
					
		#Sync the threads 
		start_sync_threads = time.time()
		
		for thread in threads:
			thread.join()
			
		sync_threads_time += time.time() - start_sync_threads
			
		print("Total time: %f Traintime: %f " % (total_time, traintime))
		save_path = saver.save(sess, MODEL_PATH)
