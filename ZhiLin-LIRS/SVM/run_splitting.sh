# On Optane SSD
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/3DXpoint/workload/SVM/webspam.train .40/;
rm /media/3DXpoint/workload/SVM/webspam.train.40/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/3DXpoint/workload/SVM/epsilon_normalized .30/;
rm /media/3DXpoint/workload/SVM/epsilon_normalized.30/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/3DXpoint/workload/SVM/kddb .40/;
rm /media/3DXpoint/workload/SVM/kddb.40/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/3DXpoint/workload/SVM/higgs .40/;
rm /media/3DXpoint/workload/SVM/higgs.40/*.sep;

# On SSD
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/ssd/workload/SVM/webspam.train .40/;
rm /media/ssd/workload/SVM/webspam.train.40/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/ssd/workload/SVM/epsilon_normalized .30/;
rm /media/ssd/workload/SVM/epsilon_normalized.30/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/ssd/workload/SVM/kddb .40/;
rm /media/ssd/workload/SVM/kddb.40/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/ssd/workload/SVM/higgs .40/;
rm /media/ssd/workload/SVM/higgs.40/*.sep;

# On Disk
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/disk/workload/SVM/webspam.train .40/;
rm /media/disk/workload/SVM/webspam.train.40/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/disk/workload/SVM/epsilon_normalized .30/;
rm /media/disk/workload/SVM/epsilon_normalized.30/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/disk/workload/SVM/kddb .40/;
rm /media/disk/workload/SVM/kddb.40/*.sep;
echo 3 > /proc/sys/vm/drop_caches;
./LIRS/splitting_time /media/disk/workload/SVM/higgs .40/;
rm /media/disk/workload/SVM/higgs.40/*.sep;