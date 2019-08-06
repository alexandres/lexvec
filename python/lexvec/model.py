# Copyright (c) 2016 Salle, Alexandre <alex@alexsalle.com>
# Author: Salle, Alexandre <alex@alexsalle.com>
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
import sys
import logging
import struct
import numpy as np

MAGIC_NUMBER = 0xbea25956
MODEL_VERSION = 1

class Model:
    def __init__(self, path):
        self._f = open(path, 'rb')
        self._parse_header()
        print('vocab_size = %d, buckets = %d, dim = %d, minn = %d, maxn = %d' % (self._vocab_size, self._buckets, self._dim, self._minn, self._maxn), file=sys.stderr)

    def _read_int(self):
        self._bytes_read += 4
        return struct.unpack('I', self._f.read(4))[0]

    def _parse_header(self):
        self._bytes_read = 0
        magic_number = self._read_int()
        if magic_number != MAGIC_NUMBER:
            raise Exception("bad magic number %d, expected %d", magic_number, MAGIC_NUMBER)
        model_version = self._read_int()
        if model_version != MODEL_VERSION:
            raise Exception("bad version %d, expected %d", model_version, MODEL_VERSION)
        self._vocab_size = self._read_int()
        self._subword_matrix_rows = self._read_int()
        self._buckets = self._subword_matrix_rows - self._vocab_size
        self._dim = self._read_int()
        self._minn = self._read_int()
        self._maxn = self._read_int()
        self._i2w = []
        self._w2i = {}
        for i in range(self._vocab_size):
            w_len = self._read_int()
            w = self._f.read(w_len).decode('utf-8')
            self._bytes_read += w_len
            self._w2i[w] = len(self._i2w)
            self._i2w.append(w)
        self._matrix_base_offset = self._bytes_read

    def _subword_idx(self, sw):
        return self._vocab_size + self._fnv(sw.encode('utf-8')) % self._buckets

    @classmethod
    def _fnv(cls, data):
        h = 0x811c9dc5
        sh = 1<<32
        cast = ord if sys.version_info[0] < 3 else lambda x: x
        for byte in data:
            h = (h * 0x01000193) % sh
            h = h ^ cast(byte)
        return h

    def _compute_subwords(self, unwrapped_w):
        w = "<%s>" % unwrapped_w
        if len(w) < self._minn:
            return []
        subwords = []
        for i in range(0, len(w)-self._minn+1):
            l = self._minn
            while l < len(w) and l <= self._maxn and i+l <= len(w):
                subwords.append(w[i:i+l])
                l += 1
        return subwords


    def _get_vector(self, idx):
        self._f.seek(self._matrix_base_offset+self._dim*idx*8)
        return np.frombuffer(self._f.read(self._dim*8), dtype=np.float64)

    def word_rep(self, w, subwords=None):
        v = np.zeros(self._dim)
        l = 0
        if w in self._w2i:
            v += self._get_vector(self._w2i[w])
            l += 1
        if self._minn > 0 and subwords is None:
            subwords = self._compute_subwords(w)
        for sw in subwords:
            v += self._get_vector(self._subword_idx(sw))
            l += 1
        if l > 0:
            v /= l
        return v

if __name__ == '__main__':
    path = sys.argv[1]
    m = Model(path)
    i = 0
    for line in sys.stdin:
        i += 1
        if i % 1000 == 0:
            print("\r%dK" % (i/1000), end="", file=sys.stderr)
        parts = line.split()
        w = parts[0]
        subwords = None
        if len(parts) > 1:
            subwords = parts[1:]
        print(w, ' '.join(map(lambda x: "%.6f" % x, m.word_rep(w, subwords).tolist())))
    print("\nfinished!", file=sys.stderr)





