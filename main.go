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
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime/pprof"
	"strings"
)

type (
	idxUint   = uint32
	countUint = uint32
	real      = float64
)

const (
	uint32Bytes  = 4
	float64Bytes = 8

	errorLogLevel = 0
	infoLogLevel  = 1
	debugLogLevel = 2

	ctxBreakToken     = "</s>"
	maxSentenceLen    = 1000
	contextPathSuffix = ".context"

	defaultProgressInterval = 10000

	vocabCommand         = "vocab"
	trainCommand         = "train"
	externalCoocCommand  = "cooc"
	externalTrainCommand = "trainem"
	oovCommand           = "embed"
)

// GLOBAL VARS

// general
var verbose int
var randng *rand.Rand

// paths
var vocabPath string
var subwordPath string
var corpusPath, coocPath, coocTotalsPath string
var vectorOutputPath string
var subvecsOutputPath string

// vocab
var corpusSize, rawCorpusSize uint64
var vocab map[string]*word
var vocabSize idxUint
var vocabList []*word
var ctxVocab map[string]*word
var ctxVocabSize idxUint
var ctxVocabList []*word
var minFreq countUint
var maxVocab idxUint
var positionalContexts, periodIsWhitespace bool
var ctxbreakw *word
var ctxbreakbytes []byte

// subword
var subwordMinN, subwordMaxN int

// cooc and sampling
var unigramPower real
var associationMeasureString string
var contextDistributionSmoothing, cdsTotal, logCdsTotal, subsample real
var window int
var weightedWindow bool
var negative int
var coocStorage matrix
var lineBufMem real
var noiseSampler sampler

// sgd
var model int
var mVec, mCtx []real
var numThreads int
var initialAlpha real
var iterations int
var dim idxUint
var subwordMatrixRows idxUint
var associationMeasure associationMeasureFunc

func init() {
	ctxbreakbytes = []byte(ctxBreakToken)
}

func main() {
	randng = rand.New(rand.NewSource(1))
	flags := flag.NewFlagSet("default", flag.ExitOnError)
	flags.StringVar(&corpusPath, "corpus", "", "path to corpus")
	flags.StringVar(&vocabPath, "vocab", "", "path where to output/load vocab")
	flags.IntVar(&subwordMinN, "minn", 3, "mininum ngram length when generating subwords")
	flags.IntVar(&subwordMaxN, "maxn", 6, "maximum ngram length when generating subwords")
	flags.StringVar(&subwordPath, "subword", "", "path to subword information")
	flags.Float64Var(&initialAlpha, "alpha", 0.025, "learning rate")
	flags.Float64Var(&subsample, "subsample", 1e-5, "subsampling threshold")
	flags.Float64Var(&contextDistributionSmoothing, "cds", 0.75, "context distribution smoothing")
	var dimRaw = flags.Int("dim", 300, "number of dimensions of word vectors")
	flags.IntVar(&iterations, "iterations", 5, "how many times to process corpus")
	flags.IntVar(&window, "window", 2, "symmetric window of (window, word, window)")
	var minFreqRaw = flags.Int("minfreq", 100, "remove from vocab words that occur less that this number of times")
	var maxVocabRaw = flags.Int("maxvocab", 0, "max vocab size, 0 for no limit")
	flags.IntVar(&negative, "negative", 5, "number of negative samples")
	flags.Float64Var(&unigramPower, "unigrampow", 0.75, "raise unigram dist to this power")
	flags.BoolVar(&weightedWindow, "weightwindow", false, "use randomized window size from uniform(1, window)")
	flags.IntVar(&model, "model", 1, "0 = output W, C; 1 = output W; 2 = output W + C")
	flags.IntVar(&numThreads, "threads", 12, "number of threads to use")
	var matrixStrings []string
	for matrix := range associationMap {
		matrixStrings = append(matrixStrings, matrix)
	}
	flags.StringVar(&associationMeasureString, "matrix", "ppmi", "which matrix to factor ("+strings.Join(matrixStrings, ",")+") default = ppmi")
	flags.IntVar(&verbose, "verbose", debugLogLevel, "verboseness (0 = errors only, 1 = info, 2 = debug) default = 1")
	flags.StringVar(&coocTotalsPath, "cooctotalspath", "", "path to cooc totals for each word when using external memory")
	flags.StringVar(&coocPath, "coocpath", "", "path to coocs when using external memory")
	flags.Float64Var(&lineBufMem, "memory", 4, "GB of memory to use in line buffer")
	flags.BoolVar(&positionalContexts, "pos", true, "use positional contexts")
	flags.StringVar(&vectorOutputPath, "output", "", "where to save vectors")
	flags.StringVar(&subvecsOutputPath, "outputsub", "", "where to save binary subword vectors")
	flags.BoolVar(&periodIsWhitespace, "periodiswhitespace", false, "treat period as whitespace")
	var cpuprofile = flags.String("cpuprofile", "", "write cpu profile to file")

	flags.Usage = func() {
		fmt.Printf("Usage: lexvec [command] [options]\n" +
			"Commands: vocab, cooc, train, trainem, embed\n" +
			"Options:\n")
		flags.PrintDefaults()
	}

	if len(os.Args) < 2 {
		flags.Usage()
		os.Exit(1)
	}
	command := os.Args[1]

	flags.Parse(os.Args[2:])

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		check(err)
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	dim = idxUint(*dimRaw)
	minFreq = countUint(*minFreqRaw)
	maxVocab = idxUint(*maxVocabRaw)

	var ok bool
	associationMeasure, ok = associationMap[associationMeasureString]
	if !ok {
		logln(errorLogLevel, "invalid association measure")
	}

	switch command {
	case vocabCommand:
		buildVocab()
		saveVocab()
	case trainCommand:
		readVocab()
		processSubwords()
		buildCoocMatrix()
		calculateCdsTotalAndLogs()
		initModel()
		train(newTrainIteratorIM())
		saveVectors()
	case externalCoocCommand:
		readVocab()
		buildCoocFile()
	case externalTrainCommand:
		readVocab()
		processSubwords()
		readCoocTotals()
		calculateCdsTotalAndLogs()
		initModel()
		train(newTrainIteratorEM())
		saveVectors()
	case oovCommand:
		calculateOovVectors()
	default:
		flags.Usage()
		os.Exit(1)
	}

	logln(infoLogLevel, "finished!")
}
