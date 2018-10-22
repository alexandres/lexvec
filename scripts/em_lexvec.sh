#!/bin/bash

set -e 
OUTPUT=${OUTPUT:?"Need to set OUTPUT"}
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN=${LEXVEC:-$DIR/../lexvec}
ARGS="-vocab $OUTPUT/vocab.txt \
    -coocpath $OUTPUT/coocs.bin \
    -cooctotalspath $OUTPUT/cooctotals.txt \
    -output $OUTPUT/vectors.txt \
    -outputsub $OUTPUT/model.bin"
mkdir -p $OUTPUT
CMD=$1
if [ $# -gt 0 ]
then
    shift
fi
if [ -z "$CMD" ] || [[ ${CMD:0:1} == "-" ]]
then
    time $BIN vocab $ARGS $CMD $@ 
    time $BIN cooc $ARGS $CMD $@
    time $BIN trainem $ARGS $CMD $@
else
    time $BIN $CMD $ARGS $@ 
fi


