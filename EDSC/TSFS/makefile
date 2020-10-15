MAINS:=$(patsubst %.cpp, bin/%, $(shell grep -l 'int main' *.cpp))

all: $(MAINS)

bin/%: %.cpp util.cpp util.h DataSetInformation.h
	g++ -std=c++11 -O3 -g $< util.cpp -o $@
