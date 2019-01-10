#!/bin/bash
for ((i = 1; i < 1000000; i++)); do
   echo 1 > /proc/sys/vm/drop_caches
   sleep 10
done
