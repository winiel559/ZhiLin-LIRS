# On Optane SSD
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/3DXpoint/workload/SVM/webspam.train;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/3DXpoint/workload/SVM/epsilon_normalized;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/3DXpoint/workload/SVM/kddb;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/3DXpoint/workload/SVM/higgs;

# On SSD
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/ssd/workload/SVM/webspam.train;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/ssd/workload/SVM/epsilon_normalized;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/ssd/workload/SVM/kddb;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/ssd/workload/SVM/higgs;

# On Disk
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/disk/workload/SVM/webspam.train;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/disk/workload/SVM/epsilon_normalized;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/disk/workload/SVM/kddb;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/preprocess_time /media/disk/workload/SVM/higgs;