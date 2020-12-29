MAINS:=$(patsubst %.cpp, bin/%, $(shell grep -l 'int main' *.cpp))

all: bin $(MAINS)

bin/%: %.cpp util.cpp util.h
	g++ -std=c++11 -O3 -g $< util.cpp -o $@

bin:
	mkdir -p bin
