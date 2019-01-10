CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a zlib/libz.a
SHVER = 1
#LIBS = -lblas

all: blocktrain blockspliter

lib: linear.o tron.o blas/blas.a
	$(CXX) -shared -dynamiclib linear.o tron.o $(LIBS) -o liblinear.so.$(SHVER)

blocktrain: tron.o linear.o blocktrain.cpp block.o $(LIBS) block.h
	$(CXX) $(CFLAGS) -o blocktrain blocktrain.cpp block.o tron.o linear.o $(LIBS)

blockspliter: blockspliter.cpp $(LIBS)
	$(CXX) $(CFLAGS) -o blockspliter blockspliter.cpp $(LIBS)

block.o: block.cpp block.h
	$(CXX) $(CFLAGS) -c -o block.o block.cpp

train: tron.o linear.o train.c $(LIBS)
	$(CXX) $(CFLAGS) -o train train.c tron.o linear.o $(LIBS)

predict: tron.o linear.o predict.c $(LIBS)
	$(CXX) $(CFLAGS) -o predict predict.c tron.o linear.o $(LIBS)

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
	cd zlib; make clean
	cd zlib-1.2.5-simple/; make clean
	rm -f *~ tron.o linear.o block.o train predict liblinear.so.$(SHVER) blocktrain blockspliter
