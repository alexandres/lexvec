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

package lexvec

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path"
	"runtime/pprof"
	"strings"
)

type OovVectors map[string][]float64
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
var buckets int

// cooc and sampling
var unigramPower real
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
var clipPmi real
var processStrategy processStrategyFunc
var processThreshold real

func init() {
	ctxbreakbytes = []byte(ctxBreakToken)
}

func Build() {
	randng = rand.New(rand.NewSource(1))
	flags := flag.NewFlagSet("default", flag.ExitOnError)
	flags.StringVar(&corpusPath, "corpus", "", "path to corpus")
	flags.StringVar(&vocabPath, "vocab", "", "path where to output/load vocab")
	flags.IntVar(&subwordMinN, "minn", 3, "mininum ngram length when generating subwords")
	flags.IntVar(&subwordMaxN, "maxn", 6, "maximum ngram length when generating subwords")
	flags.IntVar(&buckets, "buckets", 2000000, "subword buckets")
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
	var associationMeasureString = flags.String("matrix", "cpmi", "which matrix to factor ("+strings.Join(matrixStrings, ",")+")")
	flags.Float64Var(&clipPmi, "clip", 0, "clip value for cPMI (default of 0 gives PPMI) default = 0")
	var processFuncStrings []string
	for processFunc := range processStrategyMap {
		processFuncStrings = append(processFuncStrings, processFunc)
	}
	var processFuncString = flags.String("process", "all", "which matrix cells to factor ("+strings.Join(processFuncStrings, ",")+")")
	flags.Float64Var(&processThreshold, "processthreshold", 0, "threshold for -process flag (ex. 1: -process gt and -processthreshold 0 means only process cells > 0, ex. 2: -process leq and -processthreshold 0 means only process cells <= 0)")
	flags.IntVar(&verbose, "verbose", debugLogLevel, "verboseness (0 = errors only, 1 = info, 2 = debug)")
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
	associationMeasure, ok = associationMap[*associationMeasureString]
	if !ok {
		logln(errorLogLevel, "invalid option for -matrix")
	}

	processStrategy, ok = processStrategyMap[*processFuncString]
	if !ok {
		logln(errorLogLevel, "invalid option for -process")
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

func GetOovVectors(words []string, subvecsOutputPath string) ([]float64, error) {
	var (
		err              error
		b                = make([]byte, float64Bytes)
		vector           = make([]float64, 0)
		matrixBaseOffset int64
	)
	subvecsOutput, err := os.Open(subvecsOutputPath)
	if err != nil {
		logln(errorLogLevel, "open file failed")
	}
	defer subvecsOutput.Close()

	magicNumber := binaryModelReadUint32(subvecsOutput, b)
	version := binaryModelReadUint32(subvecsOutput, b)
	vocabSize := binaryModelReadUint32(subvecsOutput, b)
	subwordMatrixRows := binaryModelReadUint32(subvecsOutput, b)
	dim := binaryModelReadUint32(subvecsOutput, b)
	subwordMinN := binaryModelReadUint32(subvecsOutput, b)
	subwordMaxN := binaryModelReadUint32(subvecsOutput, b)

	if magicNumber != binaryModelMagicNumber {
		logln(errorLogLevel, "magic number doesnt match")
	}
	if version != binaryModelVersion {
		logln(errorLogLevel, "version number doesnt match")
	}

	var ivWords []string
	ivWordToIdx := make(map[string]int)
	for i := 0; i < int(vocabSize); i++ {
		wLen := binaryModelReadUint32(subvecsOutput, b)

		b := make([]byte, wLen)
		if _, err = subvecsOutput.Read(b); err != nil {
			return vector, nil
		}

		w := string(b)
		ivWordToIdx[w] = len(ivWords)
		ivWords = append(ivWords, w)
	}
	if matrixBaseOffset, err = subvecsOutput.Seek(0, 1); err != nil {
		return vector, nil
	}

	for _, oov := range words {
		vec := make([]float64, dim)
		if len(oov) == 0 {
			break
		}
		parts := strings.Split(oov, " ")
		w := parts[0]
		var subwords []string
		if subwordMinN > 0 && len(parts) == 1 {
			subwords = computeSubwords(w, int(subwordMinN), int(subwordMaxN))
		} else {
			subwords = parts[1:]
		}
		for j := 0; j < int(dim); j++ {
			vec[j] = 0
		}
		var vLen int
		if idx, ok := ivWordToIdx[w]; ok {
			sumVecFromBin(subvecsOutput, matrixBaseOffset, vec, idxUint(idx))
			vLen++
		}
		for _, sw := range subwords {
			sumVecFromBin(subvecsOutput, matrixBaseOffset, vec, subwordIdx(sw, vocabSize, subwordMatrixRows-vocabSize))
			vLen++
		}
		if vLen > 0 {
			for j := 0; j < int(dim); j++ {
				vec[j] /= float64(vLen)
			}
		}
		for _, f := range vec {
			if f != 0 {
				vector = append(vector, f)
			}
		}
	}
	return vector, nil
}

func StartTrain(outputFolder, corpusP string,
	dimP idxUint, subsampleP real,
	minfreqP countUint,
	modelP, windowP, negativeP, iterationsP, subwordMinNP int) {
	if _, err := os.Stat(outputFolder); os.IsNotExist(err) {
		_ = os.Mkdir(outputFolder, 0644)
	}

	randng = rand.New(rand.NewSource(1))
	vocabPath = path.Join(outputFolder, "vocab.txt")
	vectorOutputPath = path.Join(outputFolder, "vectors.txt")
	subvecsOutputPath = path.Join(outputFolder, "model.bin")
	corpusPath = corpusP
	dim = dimP
	subsample = subsampleP
	window = windowP
	negative = negativeP
	iterations = iterationsP
	minFreq = minfreqP
	model = modelP
	subwordMinN = subwordMinNP

	buildVocab()
	saveVocab()

	readVocab()
	processSubwords()
	buildCoocMatrix()
	calculateCdsTotalAndLogs()
	initModel()
	train(newTrainIteratorIM())
	saveVectors()
}
