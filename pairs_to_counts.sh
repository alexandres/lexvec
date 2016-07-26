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


# This script can be sped up significantly by using nsort and pypy. Uncomment lines (and comment corresponding lines) below if you have these installed.

set -e

export MI=${MI:-false}
export MEMORY=${MEMORY:-4}
TMPFILE=`mktemp`
TMPFILE2=`mktemp`

echo sorting to $TMPFILE 1>&2
sort -k1,2 -S "$MEMORY"G - | uniq -c | awk '{print $2 " " $3 " " $1 " " $4}' > $TMPFILE
# ./nsort -T"$TMPDIR" - | uniq -c | awk '{print $2 " " $3 " " $1 " " $4}' > $TMPFILE

echo removing lines ending in \* if they match previous line 1>&2
python line_merge.py < $TMPFILE > $TMPFILE2 
# pypy line_merge.py < $TMPFILE > $TMPFILE2 
rm $TMPFILE

echo shuffling $TMPFILE2 1>&2
python shuffle.py < $TMPFILE2
# pypy shuffle.py < $TMPFILE2
rm $TMPFILE2

