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

from __future__ import print_function
import argparse
import re
import sys

if sys.version_info < (3,):
    range = xrange

parser = argparse.ArgumentParser()
parser.add_argument("vectors", type=str)
args = parser.parse_args()
vectors = args.vectors
with open(vectors) as vec:
    with open(vectors + '.context') as ctx:
        with open(vectors + '.merged', 'w') as merged:
            num, d = map(lambda x: int(x), vec.readline().strip().split())
            numCtx, d2 = map(lambda x: int(x), ctx.readline().strip().split())
            print(num, d, file=merged)
            ctxVec = {}
            pos = False
            if not num == numCtx:
                pos = True
            for line in ctx:
                parts = line.strip().split()
                w = parts[0]
                if pos:
                    match = re.match("(.+)_([-0-9]+)$", parts[0])
                    w = match.group(1)
                vw = list(map(lambda x: float(x), parts[1:1 + d]))
                if w in ctxVec:
                    old = ctxVec[w]
                    for i in range(d):
                        old[i] += vw[i]
                else:
                    ctxVec[w] = vw
            not_found = 0
            for line in vec:
                parts = line.strip().split()
                w = parts[0]
                vw = list(map(lambda x: float(x), parts[1:1 + d]))
                if w in ctxVec:
                    vc = ctxVec[w]
                    for i in range(d):
                        vw[i] += vc[i]
                else:
                    not_found += 1
                print(w, ' '.join(map(lambda x: "%.6f" % (x,), vw)), file=merged)
            print("not found was", not_found)
