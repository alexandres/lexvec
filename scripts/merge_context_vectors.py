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

import numpy as np

if sys.version_info < (3,):
    range = xrange

parser = argparse.ArgumentParser()
parser.add_argument("vectors", type=str)
args = parser.parse_args()
with open(args.vectors) as vec:
    with open(args.vectors + '.context') as ctx:
        with open(args.vectors + '.merged', 'w') as merged:
            num, d = map(lambda x: int(x), vec.readline().strip().split())
            num_ctx, d_ctx = map(lambda x: int(x), ctx.readline().strip().split())
            assert d == d_ctx, "number of dimensions of word and context vectors don't match"
            print(num, d, file=merged)
            ctx_vec = {}
            pos = num < num_ctx
            for line in ctx:
                w, v_str = line.strip().split(" ", 1)
                v = np.fromstring(v_str, sep=" ") 
                if pos:
                    w = w.rpartition("_")[0]
                if w in ctx_vec:
                    ctx_vec[w] += v
                else:
                    ctx_vec[w] = v
            not_found = 0
            for line in vec:
                w, v_str = line.strip().split(" ", 1)
                v = np.fromstring(v_str, sep=" ") 
                if w in ctx_vec:
                    v += ctx_vec[w]
                else:
                    not_found += 1
                print(w, ' '.join(map(lambda x: "%.6f" % (x,), v)), file=merged)
            if not_found:
                print("failed to find %d matching context vectors" % not_found, file=sys.stderr)
