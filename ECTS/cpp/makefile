all: ects

ects: index dir
	g++ -O1 Hierarchical.cpp minValue.cpp util.cpp -o bin/ects

index: dir
	g++ -O1 IndexBuilding.cpp util.cpp -o bin/index

dir:
	mkdir -p bin

test: ects
	mkdir -p /tmp/ECG
	bin/index
	bin/ects loose
	bin/ects strict
	rm -r /tmp/ECG