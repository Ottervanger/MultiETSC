all: ects

ects: index dir
	g++ Hierarchical.cpp Euclidean.cpp minValue.cpp util.cpp -o bin/ects

index: dir
	g++ IndexBuilding.cpp Euclidean.cpp minValue.cpp -o bin/index

dir:
	mkdir -p bin

test: ects
	mkdir -p /tmp/ECG
	bin/index
	bin/ects