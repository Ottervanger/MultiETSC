rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))
empt:=
spac:= $(empt) $(empt)

SRCS = $(call rwildcard, src/, *.java)
CLSS = $(patsubst src/%.java,bin/%.class,$(SRCS))

# sourcepath ; classpath ; output path
SP = src
CP = "bin:lib/*"
OP = bin

# default
all: lib/aeatk.jar $(CLSS)

# jar
jar: all
	rm -f lib/paramils.jar
	echo "3.0.0" > src/paramils-version.txt
	cp src/paramils-version.txt bin
	cd bin && jar cvf paramils.jar . > /dev/null
	mv bin/paramils.jar lib

# aeatk
# lib/aeatk.jar: ../aeatk/aeatk.jar
# 	cp ../aeatk/aeatk.jar lib

# clean
clean:
	rm -rf bin/*
	echo "3.0.0" > src/paramils-version.txt
	rm -f lib/paramils.jar

# general file compilation rule
bin/%.class: src/%.java
	javac -sourcepath $(SP) -cp $(CP) -d $(OP) $<
