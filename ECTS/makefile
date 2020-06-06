all: bin/ects

bin/ects: bin ects.cpp util.cpp
	g++ -O3 -g ects.cpp util.cpp -o bin/ects

bin:
	mkdir -p bin

test: bin/ects
	bin/ects --version loose  --data /data/UCR/ECG200/ECG200_TRAIN.tsv /data/UCR/ECG200/ECG200_TEST.tsv
	bin/ects --version strict  --data /data/UCR/ECG200/ECG200_TRAIN.tsv /data/UCR/ECG200/ECG200_TEST.tsv
	bin/ects --version loose  --data /data/UCR/ECG200/ECG200_TRAIN.tsv /data/UCR/ECG200/ECG200_TEST.tsv --s
	bin/ects --version strict  --data /data/UCR/ECG200/ECG200_TRAIN.tsv /data/UCR/ECG200/ECG200_TEST.tsv --s
