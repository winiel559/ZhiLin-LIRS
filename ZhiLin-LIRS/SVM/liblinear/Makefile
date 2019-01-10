CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -fPIC 
LIBS = blas/blas.a zlib/libz.a
#LIBS = -lblas

all: train predict

train: tron.o linear.o train.c blas/blas.a binary.o
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o binary.o $(LIBS)

predict: tron.o linear.o predict.c blas/blas.a 
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

binary.o: binary.cpp linear.h binary.h $(LIBS)
	$(CXX) $(CFLAGS) -c -o binary.o binary.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a:
	cd blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

zlib/libz.a:
	cd zlib; make libz.a;

clean:
	cd blas;	make clean
	cd matlab;	make clean
	cd zlib; make clean
	rm -f *~ tron.o linear.o train predict binary.o
