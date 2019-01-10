#!/bin/sh
#./blocktrain -s 3 -e 0.0001 -B -1 -c 1 -m 30 -p -r 1 -t /media/ssd/workload/SVM/webspam_14G/webspam.test.cbin /media/ssd/workload/SVM/webspam_14G/ > log/web40_acc;
#./blocktrain -s 3 -e 0.0001 -B -1 -c 1 -m 30 -p -r 1 -t /media/ssd/workload/SVM/epsilon_11G/epsilon_normalized.t.cbin /media/ssd/workload/SVM/epsilon_11G/ > log/epsilon30_acc;
#./blocktrain -s 3 -e 0.0001 -B -1 -c 1 -m 30 -p -r 1 -t /media/ssd/workload/SVM/kddb_5G/kddb.t.cbin /media/ssd/workload/SVM/kddb_5G/ > log/kdd40_acc;

echo 3 > /proc/sys/vm/drop_caches;
cgexec -g memory:/mymemory ./blocktrain -s 3 -e 0.0001 -B -1 -c 1 -m 30 -p -r 1  /media/3DXpoint/workload/SVM/webspam_14G/ > log/web40_time;

echo 3 > /proc/sys/vm/drop_caches;
cgexec -g memory:/mymemory ./blocktrain -s 3 -e 0.0001 -B -1 -c 1 -m 30 -p -r 1  /media/3DXpoint/workload/SVM/epsilon_11G/ > log/epsilon30_time;

echo 3 > /proc/sys/vm/drop_caches;
cgexec -g memory:/mymemory ./blocktrain -s 3 -e 0.0001 -B -1 -c 1 -m 30 -p -M 100 -r 1  /media/3DXpoint/workload/SVM/kddb_5G/ > log/kdd40_time;
