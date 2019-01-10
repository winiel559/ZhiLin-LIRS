echo 3 > /proc/sys/vm/drop_caches;
strace -c cgexec -g memory:/mymemory ./LIRS/blocktrain -e 0.0001 -B -1 -c 1 -m 1 -p -r 1 -s 3 /media/3DXpoint/workload/SVM/kddb_5G/ >& kdd_random_page_profile;

echo 3 > /proc/sys/vm/drop_caches;
strace -c cgexec -g memory:/mymemory ./LIRS/blocktrain -e 0.0001 -B -1 -c 1 -m 1 -p -r 0 -s 3 /media/3DXpoint/workload/SVM/kddb_5G/ >& kdd_sequential_profile;

echo 3 > /proc/sys/vm/drop_caches;
strace -c cgexec -g memory:/mymemory ./LIRS/blocktrain -e 0.0001 -B -1 -c 1 -m 1 -p -r 1 -s 3 /media/3DXpoint/workload/SVM/higgs_7G/ >& higgs_random_page_profile;

echo 3 > /proc/sys/vm/drop_caches;
strace -c cgexec -g memory:/mymemory ./LIRS/blocktrain -e 0.0001 -B -1 -c 1 -m 1 -p -r 0 -s 3 /media/3DXpoint/workload/SVM/higgs_7G/ >& higgs_sequential_profile;

#If you want to dump all system calls, take off the '-c' option.