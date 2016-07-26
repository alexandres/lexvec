#!/bin/bash

# Copyright (c) 2016 Salle, Alexandre <atsalle@inf.ufrgs.br>
# Author: Salle, Alexandre <atsalle@inf.ufrgs.br>
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

set -e

if [ -z "$OUTPUTDIR" ]; then echo "Need to set OUTPUTDIR"; fi

mkdir -p $OUTPUTDIR

export TMPDIR=$OUTPUTDIR

export MI=${MI:-false}
COOC=$OUTPUTDIR/coocs
COOC_TOTALS=$COOC.totals
VOCAB=$OUTPUTDIR/vocab

CMD="./lexvec $@ -mi=$MI -cooctotalspath $COOC_TOTALS -externalmemory"

echo identifying w,c pairs
eval $CMD -printcooc -coocpath $COOC -savevocab $VOCAB 

echo aggregating pairs
./pairs_to_counts.sh < $COOC > $COOC.ready
rm $COOC

echo traning model
eval $CMD -coocpath $COOC.ready -output $OUTPUTDIR/vectors -readvocab $VOCAB
