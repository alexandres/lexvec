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
	"bufio"
	"fmt"
	"math"
	"os"
	"strings"
)

const binaryModelMagicNumber = 0xbea25956
const binaryModelVersion = 1

func initModel() {

	mVec = make([]real, subwordMatrixRows*dim)
	mCtx = make([]real, ctxVocabSize*dim)

	logln(infoLogLevel, "create vectors")
	for j := idxUint(0); j < subwordMatrixRows; j++ {
		for k := idxUint(0); k < dim; k++ {
			mVec[j*dim+k] = (randng.Float64() - 0.5) / real(dim)
		}
	}

	for j := idxUint(0); j < ctxVocabSize; j++ {
		for k := idxUint(0); k < dim; k++ {
			mCtx[j*dim+k] = (randng.Float64() - 0.5) / real(dim)
		}
	}
}

func saveVectors() {
	if subvecsOutputPath != "" {
		saveBinaryModel()
	}

	// do this after saving binary model so binary model has pure word vectors
	finalizeVectors()

	// Model 2 means we add the context vector to the word vector.
	if model == 2 {
		mergeContextVectors()
	}

	logln(infoLogLevel, "outputting vectors")
	vectorOutput, err := os.Create(vectorOutputPath)
	check(err)
	defer vectorOutput.Close()
	outputStream := bufio.NewWriter(vectorOutput)
	writeVectors(outputStream, vocabList, mVec)
	outputStream.Flush()

	// Model 0 means we also output the context vectors.
	if model == 0 {
		ctxOutput, err := os.Create(vectorOutputPath + contextPathSuffix)
		defer ctxOutput.Close()
		check(err)
		ctxOutputStream := bufio.NewWriter(ctxOutput)
		writeVectors(ctxOutputStream, ctxVocabList, mCtx)
		ctxOutputStream.Flush()
	}
}

func finalizeVectors() {
	for _, w := range vocabList {
		for j := idxUint(0); j < dim; j++ {
			// j'th component of vector
			var v real
			zSize := real(len(w.subwords))
			for _, sw := range w.subwords {
				v += mVec[sw*dim+j]
			}
			mVec[w.idx*dim+j] = v / zSize
		}
	}
}

func mergeContextVectors() {
	for _, w := range vocabList {
		for j := idxUint(0); j < dim; j++ {
			var v float64
			// If we're not using positional contexts it's simple addition.
			if !positionalContexts {
				c, _ := ctxVocab[w.w]
				v = mCtx[c.idx*dim+j]
			} else {
				// We need to sum all of the positional context vectors (one for each
				// position).
				for k := -window; k <= window; k++ {
					if k == 0 {
						continue
					}
					posC := w.posW(k)
					c, _ := ctxVocab[posC]
					v += mCtx[c.idx*dim+j]
				}
			}
			mVec[w.idx*dim+j] += v
		}
	}
}

func writeVectors(w *bufio.Writer, vocabList []*word, m []float64) {
	// Write "vocabsize dim" as first line of vector file.
	fmt.Fprintf(w, "%d %d\n", len(vocabList), dim)
	for _, mapw := range vocabList {
		w.WriteString(mapw.w)
		for j := idxUint(0); j < dim; j++ {

			fmt.Fprintf(w, " %f", m[mapw.idx*dim+j])
		}
		w.WriteString("\n")
	}
}

func saveBinaryModel() {
	logln(infoLogLevel, "saving binary model")
	subvecsOutput, err := os.Create(subvecsOutputPath)
	defer subvecsOutput.Close()
	check(err)
	subvecsOutputStream := bufio.NewWriter(subvecsOutput)
	b := make([]byte, float64Bytes)
	binaryModelWriteUint32(subvecsOutputStream, b, binaryModelMagicNumber)
	binaryModelWriteUint32(subvecsOutputStream, b, binaryModelVersion)
	binaryModelWriteUint32(subvecsOutputStream, b, vocabSize)
	binaryModelWriteUint32(subvecsOutputStream, b, subwordMatrixRows)
	binaryModelWriteUint32(subvecsOutputStream, b, dim)
	binaryModelWriteUint32(subvecsOutputStream, b, uint32(subwordMinN))
	binaryModelWriteUint32(subvecsOutputStream, b, uint32(subwordMaxN))
	// write vocab
	for _, w := range vocabList {
		binaryModelWriteUint32(subvecsOutputStream, b, uint32(len(w.w)))
		_, err = subvecsOutputStream.Write([]byte(w.w))
		check(err)
	}
	for i := idxUint(0); i < subwordMatrixRows; i++ {
		for j := idxUint(0); j < dim; j++ {
			byteOrder.PutUint64(b, math.Float64bits(mVec[i*dim+j]))
			_, err := subvecsOutputStream.Write(b)
			check(err)
		}
	}
	subvecsOutputStream.Flush()
}

func binaryModelWriteUint32(w *bufio.Writer, b []byte, v uint32) {
	byteOrder.PutUint32(b, v)
	_, err := w.Write(b[:uint32Bytes])
	check(err)
}

func binaryModelReadUint32(f *os.File, b []byte) uint32 {
	_, err := f.Read(b[:uint32Bytes])
	check(err)
	return byteOrder.Uint32(b[:uint32Bytes])
}

func calculateOovVectors() {
	logln(infoLogLevel, "loading binary model")
	subvecsOutput, err := os.Open(subvecsOutputPath)
	defer subvecsOutput.Close()
	check(err)
	b := make([]byte, float64Bytes)

	magicNumber := binaryModelReadUint32(subvecsOutput, b)
	if magicNumber != binaryModelMagicNumber {
		logln(errorLogLevel, "magic number doesnt match")
	}
	version := binaryModelReadUint32(subvecsOutput, b)
	if version != binaryModelVersion {
		logln(errorLogLevel, "version number doesnt match")
	}
	vocabSize := binaryModelReadUint32(subvecsOutput, b)
	subwordMatrixRows := binaryModelReadUint32(subvecsOutput, b)
	dim := binaryModelReadUint32(subvecsOutput, b)
	subwordMinN := binaryModelReadUint32(subvecsOutput, b)
	subwordMaxN := binaryModelReadUint32(subvecsOutput, b)

	var ivWords []string
	ivWordToIdx := make(map[string]int)
	for i := 0; i < int(vocabSize); i++ {

		wLen := binaryModelReadUint32(subvecsOutput, b)

		b := make([]byte, wLen)
		_, err := subvecsOutput.Read(b)
		check(err)
		w := string(b)
		ivWordToIdx[w] = len(ivWords)
		ivWords = append(ivWords, w)
	}
	matrixBaseOffset, err := subvecsOutput.Seek(0, 1)
	check(err)

	s := bufio.NewScanner(os.Stdin)
	s.Split(bufio.ScanLines)
	logln(infoLogLevel, "reading oov words")
	v := make([]float64, dim)
	out := bufio.NewWriter(os.Stdout)
	pp := newProgressPrinter(1000)
	for s.Scan() {
		pp.inc()
		line := s.Text()
		if len(line) == 0 {
			break
		}
		parts := strings.Split(line, " ")
		w := parts[0]
		var subwords []string
		if subwordMinN > 0 && len(parts) == 1 {
			subwords = computeSubwords(w, int(subwordMinN), int(subwordMaxN))
		} else {
			subwords = parts[1:]
		}
		for j := 0; j < int(dim); j++ {
			v[j] = 0
		}
		var vLen int
		if idx, ok := ivWordToIdx[w]; ok {
			sumVecFromBin(subvecsOutput, matrixBaseOffset, v, idxUint(idx))
			vLen++
		}
		for _, sw := range subwords {
			sumVecFromBin(subvecsOutput, matrixBaseOffset, v, subwordIdx(sw, vocabSize, subwordMatrixRows-vocabSize))
			vLen++
		}
		if vLen > 0 {
			for j := 0; j < int(dim); j++ {
				v[j] /= float64(vLen)
			}
		}
		_, err = out.WriteString(w)
		check(err)
		for i := range v {
			_, err = fmt.Fprintf(out, " %f", v[i])
			check(err)
		}
		_, err = fmt.Fprintln(out)
		check(err)
		err = out.Flush()
		check(err)
	}
}

func sumVecFromBin(f *os.File, base int64, v []float64, idx idxUint) {
	dim := len(v)
	_, err := f.Seek(base+int64(dim)*int64(idx)*float64Bytes, 0)
	check(err)
	b := make([]byte, dim*float64Bytes)
	_, err = f.Read(b)
	check(err)
	for i := range v {
		z := math.Float64frombits(byteOrder.Uint64(b[float64Bytes*i : float64Bytes*(i+1)]))
		v[i] += z
	}
}
