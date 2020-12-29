# all items to be build
TEASER:=algorithms/TEASER/build/libs/TEASERrunner.jar
ECEC:=algorithms/TEASER/build/libs/ECECRunner.jar
ECTS:=algorithms/ECTS/bin/ects
EDSC:=algorithms/EDSC/bin/edsc

all: build

build: $(TEASER) $(ECEC) $(ECTS) $(EDSC) python R

algorithms/TEASER/build/libs/%:
	@cd algorithms/TEASER/ && ./gradlew jar

$(ECTS):
	make -C algorithms/ECTS

$(EDSC):
	make -C algorithms/EDSC

.PHONY: clean python R

clean:
	rm -rf algorithms/ECTS/bin
	rm -rf algorithms/EDSC/bin
	@cd algorithms/TEASER/ && ./gradlew clean

# Python required for EARLIEST, Fixed and SMAC
ifeq ($(shell which python),)
$(error "Error: could not find python")
endif
python:
	python build-util/dependencies.py

# R required for SR-CF and ECDIRE
ifeq (, $(shell which Rscript))
$(error "Error: could not find Rscript")
endif
R:
	Rscript build-util/dependencies.R

# octave required for RelClass
ifeq (, $(shell which octave))
$(error "Error: could not find octave")
endif
