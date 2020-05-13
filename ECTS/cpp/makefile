all: ects

ects: index
	g++ Hierarchical.cpp Euclidean.cpp minValue.cpp util.cpp -o bin/ects

index:
	g++ IndexBuilding.cpp Euclidean.cpp minValue.cpp -o bin/index

test: ects
	bin/index
	bin/ects
	rm ECG/DisArray
	rm ECG/Index_Train
	rm ECG/result.txt