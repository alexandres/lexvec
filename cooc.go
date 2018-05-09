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
	"container/heap"
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"unsafe"
)

var byteOrder binary.ByteOrder = binary.LittleEndian

func buildCoocMatrix() {
	logln(debugLogLevel, "initializing cooc storage")

	coocStorage = newDictMatrix(0., vocabSize, ctxVocabSize)

	logln(infoLogLevel, "identify coocurrence")

	corpus := openCorpus()
	defer corpus.Close()
	s := createScanner(corpus)

	pp := newProgressPrinter(defaultProgressInterval)

	windower(s, randng, false, func(target, mapc *word, pos int) bool {
		pp.inc()

		target.totalCooc++
		checkCountIncOverflow(target.totalCooc)
		mapc.totalCooc++
		checkCountIncOverflow(mapc.totalCooc)

		coocStorage.set(target.idx, mapc.idx, coocStorage.get(target.idx, mapc.idx)+1)
		return true
	})
}

type coocLine struct {
	wIdx    idxUint
	cIdx    idxUint
	cooc    idxUint
	sampled idxUint
	fIdx    idxUint // used for kway merge sort
}

// SortCoocLine allows sort.Sort to multi-key (widx, cidx) sort coocLines for later merging
type SortCoocLine []coocLine

func (a SortCoocLine) Len() int      { return len(a) }
func (a SortCoocLine) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a SortCoocLine) Less(i, j int) bool {
	if a[i].wIdx < a[j].wIdx {
		return true
	}
	if a[i].wIdx > a[j].wIdx {
		return false
	}
	return a[i].cIdx < a[j].cIdx
}

// Push for heap.Push
func (a *SortCoocLine) Push(x interface{}) {

	*a = append(*a, x.(coocLine))
}

// Pop for heap.Pop
func (a *SortCoocLine) Pop() interface{} {
	old := *a
	n := len(old)
	x := old[n-1]
	*a = old[0 : n-1]
	return x
}

func packCoocLine(l coocLine, b []byte) {
	f := byteOrder.PutUint32
	offset := 0
	f(b[offset:offset+uint32Bytes], l.wIdx)
	offset += uint32Bytes
	f(b[offset:offset+uint32Bytes], l.cIdx)
	offset += uint32Bytes
	f(b[offset:offset+uint32Bytes], l.cooc)
	offset += uint32Bytes
	f(b[offset:offset+uint32Bytes], l.sampled)
}

func unpackCoocLine(b []byte) coocLine {
	f := byteOrder.Uint32
	l := coocLine{}
	offset := 0
	l.wIdx = f(b[offset : offset+uint32Bytes])
	offset += uint32Bytes
	l.cIdx = f(b[offset : offset+uint32Bytes])
	offset += uint32Bytes
	l.cooc = f(b[offset : offset+uint32Bytes])
	offset += uint32Bytes
	l.sampled = f(b[offset : offset+uint32Bytes])
	return l
}

func readCoocLine(r *bufio.Reader, b []byte, tight bool) (l coocLine, err error) {
	bytes := 16
	if tight {
		bytes = 12
	}
	err = readBytes(r, b, bytes)
	if err != nil {
		return
	}
	l = unpackCoocLine(b)
	return
}

func readBytes(r *bufio.Reader, b []byte, bytes int) (err error) {
	var bytesRead, n int
	for bytesRead < bytes {
		n, err = r.Read(b[bytesRead:bytes])
		if err != nil {
			return
		}
		bytesRead += n
	}
	return
}

type coocLineMergeShuffler struct {
	buf         []coocLine
	b           []byte
	files       []*os.File
	shufFiles   []*os.File
	linesPushed uint64
}

func newCoocLineMergeShuffler() *coocLineMergeShuffler {
	sizeOfLine := uint64(unsafe.Sizeof(coocLine{}))
	linesInBuf := uint64(lineBufMem*1e9) / sizeOfLine
	coocLineBuf := make([]coocLine, 0, linesInBuf)
	return &coocLineMergeShuffler{coocLineBuf, make([]byte, unsafe.Sizeof(coocLine{})-unsafe.Sizeof(coocLine{}.fIdx)), nil, nil, 0}
}

func (f *coocLineMergeShuffler) push(w, c, cooc idxUint) {
	f.linesPushed++
	f.buf = append(f.buf, coocLine{w, c, cooc, 1, 0})
	if len(f.buf) == cap(f.buf) {
		f.flush()
	}
}

func (f *coocLineMergeShuffler) done() {
	f.flush()
	f.mergeShuffle()
}

func (f *coocLineMergeShuffler) filePath(fIdx int) string {
	return fmt.Sprintf("%s.%d", coocPath, fIdx)
}

func (f *coocLineMergeShuffler) flush() {
	if len(f.buf) == 0 {
		return
	}
	sort.Sort(SortCoocLine(f.buf))
	t, err := os.Create(f.filePath(len(f.files)))
	check(err)
	f.files = append(f.files, t)
	w := bufio.NewWriter(t)
	for i, l := range f.buf {
		if i < len(f.buf)-1 && l.wIdx == f.buf[i+1].wIdx && l.cIdx == f.buf[i+1].cIdx {
			f.buf[i+1].cooc += l.cooc
			f.buf[i+1].sampled += l.sampled
		} else {
			packCoocLine(l, f.b)
			_, err := w.Write(f.b)
			check(err)
		}
	}
	w.Flush()
	f.buf = f.buf[:0]
}

func (f *coocLineMergeShuffler) mergeShuffle() {
	logln(debugLogLevel, "total lines = %d, GB = %f", f.linesPushed, float64(12*f.linesPushed)/1e9)
	f.linesPushed = 0
	logln(debugLogLevel, "have %d tmp files", len(f.files))
	var h SortCoocLine
	heap.Init(&h)
	var r []*bufio.Reader
	for fIdx, t := range f.files {
		t.Seek(0, 0)
		r = append(r, bufio.NewReader(t))
		l, err := readCoocLine(r[fIdx], f.b, false)
		check(err)
		l.fIdx = idxUint(fIdx)
		heap.Push(&h, l)

	}
	var linesRead uint64
	var previousLine coocLine
	pp := newProgressPrinter(defaultProgressInterval)
	for len(h) > 0 {
		pp.inc()
		l := heap.Pop(&h).(coocLine)
		if linesRead > 0 {
			if l.wIdx == previousLine.wIdx && l.cIdx == previousLine.cIdx {
				l.cooc += previousLine.cooc
				l.sampled += previousLine.sampled
			} else {
				f.spitLine(previousLine)
			}
		}
		previousLine = l
		linesRead++
		l, err := readCoocLine(r[previousLine.fIdx], f.b, false)
		if err == nil {
			l.fIdx = previousLine.fIdx
			heap.Push(&h, l)
		}
	}
	if linesRead > 0 {
		f.spitLine(previousLine)
	}
	f.shuffleFlush()
	f.linesPushed = 0
	for fIdx, t := range f.files {
		t.Close()
		err := os.Remove(f.filePath(fIdx))
		check(err)
	}
	logln(debugLogLevel, "have %d shuf files", len(f.shufFiles))
	f.mergeShuffled()
	for fIdx, t := range f.shufFiles {
		t.Close()
		err := os.Remove(f.filePath(fIdx) + ".shuf")
		check(err)
	}
	logln(debugLogLevel, "total lines shuffled = %d", f.linesPushed)
}

func (f *coocLineMergeShuffler) spitLine(l coocLine) {
	for k := countUint(0); k < l.sampled; k++ {
		f.buf = append(f.buf, l)
		if len(f.buf) == cap(f.buf) {
			f.shuffleFlush()
		}
	}
}

func (f *coocLineMergeShuffler) shuffleFlush() {
	if len(f.buf) == 0 {
		return
	}
	t, err := os.Create(f.filePath(len(f.shufFiles)) + ".shuf")
	check(err)
	f.shufFiles = append(f.shufFiles, t)
	w := bufio.NewWriter(t)
	f.shuffleFlushToWriter(w)
	w.Flush()
}

func (f *coocLineMergeShuffler) shuffleFlushToWriter(w *bufio.Writer) {
	if len(f.buf) == 0 {
		return
	}
	for i := len(f.buf) - 1; i >= 0; i-- {
		if i > 0 {
			j := rand.Intn(i)
			f.buf[i], f.buf[j] = f.buf[j], f.buf[i]
		}
		packCoocLine(f.buf[i], f.b)

		_, err := w.Write(f.b[:12]) // exclude CoocLine.sampled
		f.linesPushed++
		check(err)
	}
	f.buf = f.buf[:0]
	w.Flush()
}

func (f *coocLineMergeShuffler) mergeShuffled() {
	coocStream, err := os.Create(coocPath)
	check(err)
	coocStreamWriter := bufio.NewWriter(coocStream)
	var r []*bufio.Reader
	for _, t := range f.shufFiles {
		_, err := t.Seek(0, 0)
		check(err)
		r = append(r, bufio.NewReader(t))
	}
	readSomething := true
	pp := newProgressPrinter(defaultProgressInterval)
	for readSomething {
		readSomething = false
		for i, t := range r {
			if t == nil {
				continue
			}
			for j := 0; j < cap(f.buf)/len(r); j++ {
				l, err := readCoocLine(t, f.b, true)
				if err == nil {
					pp.inc()
					readSomething = true
					f.buf = append(f.buf, l)
					if len(f.buf) == cap(f.buf) {
						f.shuffleFlushToWriter(coocStreamWriter)
					}
				} else {
					r[i] = nil
					break
				}
			}
		}
	}
	f.shuffleFlushToWriter(coocStreamWriter)
	coocStreamWriter.Flush()
	coocStream.Close()
}

func buildCoocFile() {
	if len(coocPath) == 0 || len(coocTotalsPath) == 0 {
		logln(errorLogLevel, "FATAL ERROR: coocpath and cooctotalspath are required arguments")
		os.Exit(1)
	}

	var totalCoocs uint64
	for _, c := range ctxVocabList {
		totalCoocs += uint64(c.freq)
	}
	lineEstimateG := (float64(totalCoocs) + float64(negative)*float64(totalCoocs)/float64(2*window)) / 1e9
	logln(infoLogLevel, "estimate: this should use a peak of %f GB of hard disk space, and %f GB when complete, %.2fG lines", (16+12)*lineEstimateG, 12*lineEstimateG, lineEstimateG)

	corpus := openCorpus()
	defer corpus.Close()
	logln(infoLogLevel, "identify coocurrence")

	s := createScanner(corpus)

	flusher := newCoocLineMergeShuffler()

	pp := newProgressPrinter(defaultProgressInterval)
	windower(s, randng, true, func(target, mapc *word, pos int) bool {

		if mapc != nil {
			pp.inc()

			target.totalCooc++
			checkCountIncOverflow(target.totalCooc)
			mapc.totalCooc++
			checkCountIncOverflow(mapc.totalCooc)

			flusher.push(target.idx, mapc.idx, 1)
		} else {
			for k := 0; k < negative; k++ {
				mapc := noiseSampler.sample(randng)
				if mapc == target {
					continue
				}
				for mapc == ctxbreakw {
					mapc = noiseSampler.sample(randng)
				}
				pp.inc()
				flusher.push(target.idx, mapc.idx, 0)
			}
		}
		return true
	})
	flusher.done()
	writeCoocTotals(vocabList, coocTotalsPath)
	writeCoocTotals(ctxVocabList, coocTotalsPath+contextPathSuffix)
}

func writeCoocTotals(vocabList []*word, path string) {
	coocTotalsOutput, err := os.Create(path)
	check(err)
	logln(infoLogLevel, "writing cooc totals")
	for _, w := range vocabList {
		fmt.Fprintf(coocTotalsOutput, "%s %d\n", w.w, w.totalCooc)
	}
	coocTotalsOutput.Close()
}

func readCoocTotalsFile(vocab map[string]*word, path string) {
	logln(infoLogLevel, "reading cooc totals")
	coocTotalsStream, err := os.Open(path)
	check(err)
	readCounts(createScanner(coocTotalsStream), func(w string, cnt countUint) {
		mapw, ok := vocab[w]
		if !ok {
			logln(errorLogLevel, "word %s not found in %s", w, path)
		}
		mapw.totalCooc = cnt
	})
	coocTotalsStream.Close()
}

func readCoocTotals() {
	readCoocTotalsFile(vocab, coocTotalsPath)
	readCoocTotalsFile(ctxVocab, coocTotalsPath+contextPathSuffix)
}

func calculateCdsTotalAndLogs() {

	for _, w := range ctxVocabList {
		cdsTotal += math.Pow(real(w.totalCooc), contextDistributionSmoothing)
	}
	logln(infoLogLevel, "cds total: %f", cdsTotal)
	logCdsTotal = math.Log(cdsTotal)
	for _, w := range vocabList {
		w.logTotalCooc = math.Log(real(w.totalCooc))
	}
	for _, w := range ctxVocabList {
		w.logTotalCooc = math.Log(math.Pow(real(w.totalCooc), contextDistributionSmoothing))
	}
}
