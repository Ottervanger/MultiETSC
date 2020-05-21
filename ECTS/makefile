all: bin/ects

bin/ects: bin ects.cpp util.cpp
	g++ -O3 -g ects.cpp util.cpp -o bin/ects

bin:
	mkdir -p bin

test: bin/ects
	bin/ects --version loose  --path /data/UCR/ --dataset ECG200
	bin/ects --version strict --path /data/UCR/ --dataset ECG200