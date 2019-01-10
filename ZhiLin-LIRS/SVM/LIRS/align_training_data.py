import numpy as np
import random, sys, os, math

#Count max feature number
def count_max(training_data):

	file = open(training_data,'r')
	max_count = 0
	min_count = math.inf
	instance_count = 0
	
	for line in file.readlines():
		data = line.strip().split(' ')
		feature_count = len(data) - 1
		
		if feature_count > max_count:
			max_count = feature_count
		
		if feature_count < min_count:
			min_count = feature_count
			
		instance_count = instance_count + 1
	
	print("instance_count:", instance_count, ", max_count:", max_count, ", min_count:", min_count)
	
	return max_count, instance_count, min_count

#Inout 3:20 return 3
def datatoindex(data):
	temp = data.strip().split(':')
	return int(temp[0])

#Read training data
def load_train(training_data, result_dir):

	print("Start count max feature number ...")
	max_count, instance_count, min_count = count_max(training_data)
	
	print("Start reading", training_data, ", writing result to", result_dir, "...")
	original_file = open(training_data,'r')
	target_file = open(result_dir,'w')
	
	count = 0
	
	for line in original_file.readlines():
		count = count + 1
		if(count % (instance_count/10) == 0):
			print("(%d/%d) completed." % (count,instance_count))
			
		data = line.strip().split(' ')
		
		tail_data = data[len(data)-1].strip().split(':')
		#print(tail_data)
		
		feature_count = len(data) - 1
		remain_count = max_count - feature_count
		
		i = 1
		while(remain_count != 0):
			if(i==1 and datatoindex(data[i])!=1):
				insert_data = "1:0"
				data.insert(1, insert_data)
				remain_count = remain_count - 1
			
			
			try: datatoindex(data[i])
			except IndexError:
				insert_data = str(i)+":0"
				data.append(insert_data)
				remain_count = remain_count - 1
				
			if(i!=1 and datatoindex(data[i])-datatoindex(data[i-1])!=1):
				insert_data = str(i)+":0"
				data.insert(i, insert_data)
				remain_count = remain_count - 1

			i = i+1		
					
		for i in range(len(data)):
			if i == max_count:
				target_file.write("%s" % (data[i]))
			else:
				target_file.write("%s " % (data[i]))

		target_file.write("\n")
		
	original_file.close()
	target_file.close()


#
def main():
	
	load_train(sys.argv[1], sys.argv[2])
	
	
if __name__ == "__main__":
	main()
	
