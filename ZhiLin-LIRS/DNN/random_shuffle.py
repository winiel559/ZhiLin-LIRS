import sys, struct, string, os, time, random
import numpy as np

def read_image(index, f):
	
	image_offset = 0 + index*(256*256*3)
	f.seek(image_offset, 0)
	
	buf = f.read(196608)
	im_array = np.fromstring(buf, dtype='>B')
	
	return im_array
	
def read_label(index, f):
	
	label_offset = 0 + index*(2)
	f.seek(label_offset, 0)
	
	buf = f.read(2)
	label = np.fromstring(buf, dtype='>H')
	
	return label

def append_image(im_array, file):

	for i in range(256*256*3):
		file.write(struct.pack('>1B', im_array[i]))
		
def append_label(label, file):
	
	label = np.array(label).astype(np.uint16)
	file.write(struct.pack('>1H', label[0]))
	
def read_write_binary_img(index, source, target):
	image_offset = 0 + index*(256*256*3)
	source.seek(image_offset, 0)
	buf = source.read(196608)
	
	target.write(buf)
	
def read_write_binary_label(index, source, target):
	label_offset = 0 + index*(2)
	source.seek(label_offset, 0)
	buf = source.read(2)
	
	target.write(buf)
	
	

TRAINING_INSTANCE_NUM = 100000
DEBUG_INFO = 0

	
if __name__ == "__main__":

	train_img = open('imagenet_1280K/train-images','rb')
	train_label = open('imagenet_1280K/train-labels','rb')
	
	train_img_shuffle = open('imagenet_10K_shuffle/train-images','wb')
	train_label_shuffle = open('imagenet_10K_shuffle/train-labels','wb')
	
	index = np.zeros(TRAINING_INSTANCE_NUM).astype(np.uint32)
	for i in range(TRAINING_INSTANCE_NUM):
		index[i] = i
	
	random.shuffle(index)

	
	if DEBUG_INFO:
		print("Original")
		for i in range(TRAINING_INSTANCE_NUM):
			image = read_image(i, train_img)
			label = read_label(i, train_label)
			print(image, label)
	

	start = time.time()
	for i in range(TRAINING_INSTANCE_NUM):
		read_write_binary_img(index[i], train_img, train_img_shuffle)
		read_write_binary_label(index[i], train_label, train_label_shuffle)
	finish = time.time()
	
	print("Shuffle %d instances, elapse time: %f, Avg time: %f intance/seconds." \
			% (TRAINING_INSTANCE_NUM, finish-start, TRAINING_INSTANCE_NUM/(finish-start)))
	
	
	train_img.close()
	train_label.close()
	train_img_shuffle.close()
	train_label_shuffle.close()
	
	if DEBUG_INFO:
		r_train_img_shuffle = open('imagenet_10K_shuffle/train-images','rb')
		r_train_label_shuffle = open('imagenet_10K_shuffle/train-labels','rb')
		
		print("Read after shuffle")
		for i in range(TRAINING_INSTANCE_NUM):
			image = read_image(i, r_train_img_shuffle)
			label = read_label(i, r_train_label_shuffle)
			print(image, label)
	
		
	
	
	
	
	