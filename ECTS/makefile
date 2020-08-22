all: bin/ects

bin/ects: bin ects.cpp util.cpp
	g++ -std=c++14 -O3 -g ects.cpp util.cpp -o bin/ects

bin:
	mkdir -p bin

test: bin/ects
	@for data in Coffee Herring; do for v in strict loose; do for ms in  0.0 0.01 0.3 1.0; do \
		bin/ects -v $${v}  -m $${ms}  -d ../data/$${data}_TRAIN.tsv ../data/$${data}_TEST.tsv || exit; \
	done; done; done
