/*
 * Copyright (c) 2016 Salle, Alexandre <alex@alexsalle.com>
 * Author: Salle, Alexandre <alex@alexsalle.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package main

import (
	"math"
	"math/rand"
)

func (w *word) subsampleP(t real, corpusSize uint64) real {
	if t == 0 {
		return 0
	}
	p := 1 - math.Sqrt(t/(real(w.freq)/real(corpusSize)))
	if p < 0 {
		p = 0
	}
	return p
}

func (w *word) keepP(t real, corpusSize uint64) real {
	return 1 - w.subsampleP(t, corpusSize)
}

type sampler interface {
	sample(r *rand.Rand) *word
}

// Unigram sampling with context distribution smoothing, implements Sampler interface.
// Ported from word2vec.
type unigramDist struct {
	vocab []*word
	table []int
}

func newUnigramDist(vocab []*word, tableSize int, power real) *unigramDist {
	var trainWordsPow real
	table := make([]int, tableSize)
	vocabSize := len(vocab)
	for i := 0; i < vocabSize; i++ {
		w := vocab[i]
		trainWordsPow += math.Pow(real(w.freq), power)
	}
	var i int
	d1 := math.Pow(real(vocab[i].freq), power) / trainWordsPow
	for a := 0; a < tableSize; a++ {
		table[a] = i
		if real(a)/real(tableSize) > d1 {
			i++
			d1 += math.Pow(real(vocab[i].freq), power) / trainWordsPow
		}
		if i >= vocabSize {
			i = vocabSize - 1
		}
	}
	return &unigramDist{vocab, table}
}

func (d *unigramDist) sample(r *rand.Rand) *word {
	i := r.Intn(len(d.table))
	w := d.vocab[d.table[i]]
	return w
}
