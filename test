#!/usr/bin/env bash
set -e
export TIME="  TIME: \t%E\n  CPU:  \t%P\n  MEM avg/max: \t%t/%M\n  FS I/O: \t%I/%O\n  pfaults M/m: \t%F/%R"

echo "TEST:   ECTS"
pushd ECTS > /dev/null && env time make test && echo "PASSED: ECTS" || echo "FAILED: ECTS"
popd > /dev/null

for ALG in {RelClass,ECDIRE,SR-CF}; do
	echo "TEST:   $ALG"
	pushd $ALG > /dev/null && env time ./test  && echo "PASSED: $ALG" || echo "FAILED: $ALG"
	popd > /dev/null
done
