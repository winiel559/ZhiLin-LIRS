# ZhiLin-LIRS
1. SVM	
	1. execute as root
		
		sudo su
		
	2. establish cgroup env, limit memory size
		sudo mkdir /sys/fs/cgroup/memory/mymemory
		sudo chown -R eclab:users /sys/fs/cgroup/memory/mymemory
		echo 1000000000 > /sys/fs/cgroup/memory/mymemory/memory.limit_in_bytes #limit memory size to 1GB
		
	3. make sure that you have the following 16 files and dir, in the same dir ex./media/ssd/workload/SVM/
		Raw data
			webspam.train, epsilon_normalized, kddb, higgs
		BMF splitting file
			webspam.train.40/, epsilon_normalized.30/, kddb.40/, higgs.40/
		Testing data
			webspam.test.cbin, epsilon_normalized.t.cbin, kddb.t.cbin, higgs.t.cbin
		LIRS binary file
			webspam_14G/, epsilon_11G/, kddb_5G/, higgs_7G/
		
	4. Baseline - BMF
		make sure the workload address in /cdblock/data.py is correct		
		modify /cdblock/run_BMF.py data_dir as workload's dir
		
		Loss and Accu
			./cdblock/run_BMF.py acc
		analyze performance
			./cdblock/run_BMF.py time
	5. LIRS
		make sure the workload address in /cdblock/data.py is correct		
		modify data_dir as workload's dir in /cdblock/run_BMF.py  
		Loss and Accu
			python ./cdblock/run_LIRS.py acc
		analyze performance
			python ./cdblock/run_LIRS.py time
	
	6. Run initial pre-processing time
		compile /cdblock/LIRS/preprocess_time.c, /cdblock/LIRS/splitting_time.c
			
			#LIRS
			g++ -Wall -Wconversion -O3 -fPIC -g -o preprocess_time preprocess_time.c
			
			#BMF
			g++ -Wall -Wconversion -O3 -fPIC -g -o splitting_time splitting_time.c
		
		modify workload dir in run_preprocess.sh 
		run
			./cdblock/run_preprocess.sh > pre_process.log
		
		modify workload dir in run_splitting.sh
		run
			./cdblock/run_splitting.sh > ini_splitting.log

	7. Page assignmnt vs. instance assignment

		modify workload dir in prof_syscall.sh
		./cdblock/prof_syscall.sh
	
		the original source code will see instance size and automatically use page assignment, if you want to do an experiment of instance assignment, change block.cpp line 870 to f(0), and make again, to enforce instance assignment.
		run again and modify dir of written file
		./cdblock/prof_syscall.sh
	
	8. Use your own workload
		raw train data's format must be: label index:value index:value index:value
		raw test data's format must be: label index:value index:value index:value
		(ex. 1 2:1 103:1 104:1 105:1 106:0.301 ... )
		
		Generate BMF splitting file, nBlocks is the number of batches you want to split to
			./cdblock/BMF/blockspliter -S nBlocks -c raw_train_data_dir target_dir
			(ex. ./cdblock/BMF/blockspliter -S 40 -c /media/ssd/workload/SVM/higgs /media/ssd/workload/SVM/higgs.40)
		
		Generate testing data,
			./cdblock/BMF/blockspliter -S 1 -c raw_test_data_dir test_data_dir.1
			ln -fs test_data_dir.1/data/*1.bin target.cbin
			
			(ex.
			./cdblock/BMF/blockspliter -S 1 -c /media/ssd/workload/SVM/higgs.t /media/ssd/workload/SVM/higgs.t.1
			ln -fs /media/ssd/workload/SVM/higgs.t.1/data/*1.bin /media/ssd/workload/SVM/higgs.t.cbin)
		
		Generate LIRS binary file
			compile /cdblock/LIRS/gen_binary_data.c
				g++ -Wall -Wconversion -O3 -fPIC -g -o gen_binary_data gen_binary_data.c
			./cdblock/LIRS/gen_binary_data raw_train_data_dir binary_file
			(ex. ./cdblock/LIRS/gen_binary_data /media/ssd/workload/SVM/higgs binary_file)
		
2. DNN			
	1. make sure you have imagenet
		
	2. Run Loss and Accu
		AlexNet:
			TFIP:
				python3 imagenet_alexnet.py tf_queue 128 10112 ./temp
			LIRS:
				python3 imagenet_alexnet.py random_shuffle 128 1024 ./temp
				
		Overfeat:
			TFIP:
				python3 imagenet_overfeat_accurate.py tf_queue 128 10112 ./temp
			LIRS:
				python3 imagenet_overfeat_accurate.py random_shuffle 128 1024 ./temp
		
		VGG16:
			TFIP:
				python3 ./vgg-transfer.py tf_queue 32 10016 ./temp
			LIRS:
				python3 ./vgg-transfer.py random_shuffle 32 1024 ./temp
	
	3. Run performance
	
		make sure workload dir(DATA_DIR) in the following files are correct
			files: imagenet_alexnet_performance.py, imagenet_overfeat_performance.py, /vgg-transfer/vgg-transfer-performance.py

		
		AlexNet:
			TFIP:
				python3 imagenet_alexnet_performance.py tf_queue 128 1024 ./temp
			LIRS:
				python3 imagenet_alexnet_performance.py random_shuffle 128 1024 ./temp
				
		OverFeat:
			TFIP:
				python3 imagenet_overfeat_performance.py tf_queue 128 1024 ./temp
			LIRS:
				python3 imagenet_overfeat_performance.py random_shuffle 128 1024 ./temp
				
		VGG16:
			TFIP:
				python3 ./vgg-transfer-performance.py tf_queue 32 1024 ./temp
			LIRS:
				python3 ./vgg-transfer-performance.py random_shuffle 32 1024 ./temp
	
	3. Run initial pre-processing time
	
		place random_shuffle.py and imagenet_1280K/ under the same dir
		
		mkdir imagenet_shuffle
		echo 3 > /proc/sys/vm/drop_caches;
		python3 random_shuffle.py 1281167 >> result;
		rm ./imagenet_shuffle/*;

	
	4. comparing training result with various Queue size
		
		AlexNet:
			python3 imagenet_alexnet.py fix_order 128 1024 ./temp   #Queue size = 1
			python3 imagenet_alexnet.py tf_queue 128 512 ./temp		#Queue size = 0.5K
			python3 imagenet_alexnet.py tf_queue 128 1024 ./temp	#Queue size = 1K
			python3 imagenet_alexnet.py tf_queue 128 5120 ./temp    #Queue size = 5K
			python3 imagenet_alexnet.py tf_queue 128 10112 ./temp 	#Queue size = 10K
		                                                     
		OverFeat:
			python3 imagenet_overfeat_accurate.py fix_order 128 1024 ./temp   #Queue size = 1
			python3 imagenet_overfeat_accurate.py tf_queue 128 512 ./temp     #Queue size = 0.5K
			python3 imagenet_overfeat_accurate.py tf_queue 128 1024 ./temp    #Queue size = 1K
			python3 imagenet_overfeat_accurate.py tf_queue 128 5120 ./temp    #Queue size = 5K
			python3 imagenet_overfeat_accurate.py tf_queue 128 10112 ./temp   #Queue size = 10K
		
			
			
		
		
			
