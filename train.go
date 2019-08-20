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
	"math"
	"math/rand"
	"os"
	"sync"
	"time"
)

type processStrategyFunc func(y real) bool

var processStrategyMap = map[string]processStrategyFunc{
	"all": processAllStrategyFunc,
	"gt":  processGreaterThanThresholdStrategyFunc,
	"leq": processLessThanOrEqualToThresholdStrategyFunc,
}

func processAllStrategyFunc(y real) bool {
	return true
}

func processGreaterThanThresholdStrategyFunc(y real) bool {
	return y > processThreshold
}

func processLessThanOrEqualToThresholdStrategyFunc(y real) bool {
	return y <= processThreshold
}

const batchSize = 10000

var totalLossPerThread []real
var numLossesPerThread []uint64
var datumProcessedPerThread []uint64
var alpha real
var totalDatum uint64
var iteration int

func train(it trainIterator) {
	if len(vectorOutputPath) == 0 {
		logln(errorLogLevel, "FATAL ERROR: output (where to save vectors) is a required argument")
		os.Exit(1)
	}

	logln(infoLogLevel, "running lexvec")
	// Create counters per thread to avoid synchronization.
	totalLossPerThread = make([]real, numThreads)
	numLossesPerThread = make([]uint64, numThreads)
	datumProcessedPerThread = make([]uint64, numThreads)
	totalDatum = it.totalDatum()

	sgdStepCgoInit()

	// This is used to wait on SGD threads until all are complete.
	var wg sync.WaitGroup

	for iteration = 0; iteration < iterations; iteration++ {
		for i := range totalLossPerThread {
			totalLossPerThread[i] = 0
			numLossesPerThread[i] = 0
			datumProcessedPerThread[i] = 0
		}
		for threadID := 0; threadID < numThreads; threadID++ {
			wg.Add(1)
			go trainThread(&wg, threadID, iteration, it)
		}
		quitProgressReport := make(chan bool)
		go progressReport(quitProgressReport)
		wg.Wait()
		quitProgressReport <- true
		logln(infoLogLevel, "iteration %d MSE = %f, sgdsteps %d", iteration, meanLoss(), numLosses())
	}
}

func progressReport(quit chan bool) {
	startAt := time.Now()
	for {
		select {
		case <-quit:
			return
		default:
			time.Sleep(time.Second)
			secondsElapsed := time.Now().Sub(startAt).Seconds()
			pairsPerSecond := real(numLosses()) / real(numThreads) / secondsElapsed
			currentTotalDatumProcessed := totalDatumProcessed()
			datumPerSecond := real(currentTotalDatumProcessed) / secondsElapsed
			secondsRemaining := real(uint64(iterations-iteration)*totalDatum-currentTotalDatumProcessed) / datumPerSecond
			hours := int(secondsRemaining) / 3600
			minutes := (int(secondsRemaining) % 3600) / 60
			log(debugLogLevel, "\r%4.1f%%, alpha %7.6f, speed %6.1fk pairs/thread/s, mean loss %7.6f, eta %02dh%02dm",
				progress()*1e2, alpha, pairsPerSecond/1e3, meanLoss(), hours, minutes)
		}
	}

}

func meanLoss() real {
	var totalLoss real
	var numLosses uint64
	for i, l := range totalLossPerThread {
		totalLoss += l
		numLosses += numLossesPerThread[i]
	}
	return totalLoss / real(numLosses)
}

func progress() real {
	return (real(iteration) + real(totalDatumProcessed())/real(totalDatum)) / real(iterations)
}

func numLosses() uint64 {
	var numLosses uint64
	for _, l := range numLossesPerThread {
		numLosses += l
	}
	return numLosses
}

func totalDatumProcessed() uint64 {
	var t uint64
	for _, l := range datumProcessedPerThread {
		t += l
	}
	return t
}

func trainThread(wg *sync.WaitGroup, threadID, iteration int, it trainIterator) {
	// NOW WE ARE IN AN SGD THREAD

	// Used by learn() to store gradients. Shared accross all calls in same
	// thread to avoid GC.
	zVec := make([]real, dim)
	wIdxs := make([]idxUint, batchSize)
	cIdxs := make([]idxUint, batchSize)
	ys := make([]real, batchSize)
	var n idxUint

	step := func() {
		numLossesPerThread[threadID] += uint64(n)
		totalLossPerThread[threadID] += sgdStepCgoBatch(wIdxs, cIdxs, ys, zVec, alpha, n)

		alpha = initialAlpha * (1 - progress())
		if alpha < initialAlpha*0.0001 {
			alpha = initialAlpha * 0.0001
		}
		if math.IsNaN(totalLossPerThread[threadID]) {
			logln(errorLogLevel, "\ngot a NaN, goodbye")
		}
		n = 0
	}

	it.iterate(threadID, func(w, c *word, y real, datumProcessedByThread uint64) {
		datumProcessedPerThread[threadID] = datumProcessedByThread
		if !processStrategy(y) {
			return
		}
		wIdxs[n] = w.idx
		cIdxs[n] = c.idx
		ys[n] = y
		n++
		if n == batchSize {
			step()
		}
	})
	step()
	wg.Done()
}

type trainIteratorCallback func(w, c *word, y real, datumProcessedByThread uint64)

type trainIterator interface {
	totalDatum() uint64
	iterate(threadID int, callback trainIteratorCallback)
}

type trainIteratorEM struct {
	lines int64
}

func newTrainIteratorEM() *trainIteratorEM {
	coocStat, err := os.Stat(coocPath)
	check(err)
	coocStreamFileSize := coocStat.Size()
	lines := coocStreamFileSize / 12
	return &trainIteratorEM{lines}
}

func (t *trainIteratorEM) totalDatum() uint64 {
	return uint64(t.lines)
}

func (t *trainIteratorEM) iterate(threadID int, callback trainIteratorCallback) {
	// EXTERNAL MEMORY LexVec

	coocStream, err := os.Open(coocPath)
	check(err)
	defer coocStream.Close()

	linesPerThread := t.lines / int64(numThreads)
	linesForThisThread := linesPerThread
	if threadID == numThreads-1 && t.lines%linesPerThread > 0 {
		linesForThisThread = t.lines % linesPerThread
	}
	b := make([]byte, 16)
	coocStream.Seek(int64(threadID)*linesPerThread*12, 0)
	r := bufio.NewReader(coocStream)

	// Run over each line in segment.
	for i := int64(0); i < linesForThisThread; i++ {
		l, err := readCoocLine(r, b, true)
		check(err)
		mapw := vocabList[l.wIdx]
		mapc := ctxVocabList[l.cIdx]
		y := associationMeasure(mapw, mapc, countUint(l.cooc))
		callback(mapw, mapc, y, uint64(i))
	}

}

type trainIteratorIM struct {
	corpusFileSize  int64
	randngPerThread []*rand.Rand
}

func newTrainIteratorIM() *trainIteratorIM {
	corpusStat, err := os.Stat(corpusPath)
	check(err)
	var randngPerThread []*rand.Rand
	for threadID := 0; threadID < numThreads; threadID++ {
		randngPerThread = append(randngPerThread, rand.New(rand.NewSource(int64(threadID))))
	}
	return &trainIteratorIM{corpusStat.Size(), randngPerThread}
}

func (t *trainIteratorIM) totalDatum() uint64 {
	return uint64(t.corpusFileSize)
}

func (t *trainIteratorIM) iterate(threadID int, callback trainIteratorCallback) {
	// IN-MEMORY LexVec

	// Each thread needs its own random number generator
	randn := t.randngPerThread[threadID]

	// Each thread needs handle to corpus
	corpus := openCorpus()
	defer corpus.Close()

	// Determine thread's segment within file.
	corpusOffsetStart := (t.corpusFileSize / int64(numThreads)) * int64(threadID)
	corpusOffsetEnd := (t.corpusFileSize / int64(numThreads)) * int64(threadID+1)
	if corpusOffsetEnd >= t.corpusFileSize {
		corpusOffsetEnd = t.corpusFileSize
	}
	_, err := corpus.Seek(corpusOffsetStart, 0)
	check(err)

	s := createScanner(corpus)

	// Iterate over every token in the corpus within thread's segment.
	windower(s, randn, true, func(target, mapc *word, pos int) bool {
		// This funky Seek call is to get current position within file.
		curPos, err := corpus.Seek(0, 1)
		check(err)

		// Check to see if overstepping segment. If so done with iteration.
		if curPos > corpusOffsetEnd {
			return false
		}

		datumProcessedByThread := uint64(curPos - corpusOffsetStart)
		if mapc != nil {
			callback(target, mapc, associationMeasure(target, mapc, coocStorage.get(target.idx, mapc.idx)), datumProcessedByThread)
		} else {
			for k := 0; k < negative; k++ {
				// Draw negative sample.
				mapc := noiseSampler.sample(randn)

				// Skip if sample is target word.
				if target == mapc {
					continue
				}

				// Draw until sample is not context break.
				for mapc == ctxbreakw {
					mapc = noiseSampler.sample(randn)
				}

				callback(target, mapc, associationMeasure(target, mapc, coocStorage.get(target.idx, mapc.idx)), datumProcessedByThread)
			}
		}
		return true
	})
}
