/*
 * Copyright (c) 2016 Salle, Alexandre <atsalle@inf.ufrgs.br>
 * Author: Salle, Alexandre <atsalle@inf.ufrgs.br>
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
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	ERROR = 0
	INFO  = 1
	DEBUG = 2

	PPMI_MATRIX     = "ppmi"
	PMI_MATRIX      = "pmi"
	COOC_MATRIX     = "cooc"
	LOG_COOC_MATRIX = "logcooc"

	CTXBREAK            = "</s>"
	MAX_SENTENCE_LENGTH = 1000
	CONTEXT_SUFFIX      = ".context"
)

// This is to guide the main routine as to what is being done.
type WhichLexVec int

const (
	PROC_IN_MEMORY WhichLexVec = iota
	PROC_EXTERNAL_MEMORY_PRE
	PROC_EXTERNAL_MEMORY_SGD
)

var verbose int
var dim uint32
var corpusSize, rawCorpusSize uint64
var mVec, mCtx, bVec, bCtx, mVecGrad, mCtxGrad, bVecGrad, bCtxGrad []float64
var contextDistributionSmoothing, cdsTotal, postSubsample float64
var useBias, adagrad, externalMemory, positionalContexts, periodIsWhitespace bool
var randng *rand.Rand
var matrix string
var window int
var ctxbreakw *Word
var processing WhichLexVec

var ctxbreakbytes []byte

func init() {
	ctxbreakbytes = []byte(CTXBREAK)
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Helper for aborting on error.
func check(e error) {
	if e != nil {
		panic(e)
	}
}

// Helper for logging. Should use stdlib logging instead!
func logit(msg string, lineBreak bool, level int) {
	if verbose < level {
		return
	}
	if lineBreak {
		fmt.Fprintf(os.Stderr, "\n")
	}
	fmt.Fprintf(os.Stderr, msg)
	if lineBreak {
		fmt.Fprintf(os.Stderr, "\n")
	}
	os.Stderr.Sync()
}

type Word struct {
	w         string
	i         uint32
	freq      uint64
	totalCooc float64
}

// Adds position within window to word.
// ex: i walked the dog -> target = the -> i_-2 walked_-1 the dog_1
func (w *Word) posW(pos int) string {
	return fmt.Sprintf("%s_%d", w.w, pos)
}

// The following *Direct functions calculate association measures
// between word pairs (w, c). cooc is the number of coocurrences
// of the pair. If you wish to perform a different transformation
// of the coocurrence matrix, implement a function that satisfies
// this interface.
func (w *Word) PpmiDirect(c *Word, cooc float64) float64 {
	ppmi := math.Log(cooc) - math.Log(w.totalCooc) - math.Log(math.Pow(c.totalCooc, contextDistributionSmoothing)) + math.Log(cdsTotal)
	if ppmi < 0 {
		return 0
	}
	return ppmi
}

func (w *Word) PmiDirect(c *Word, cooc float64) float64 {
	if cooc < 1 {
		cooc = 1 // smoothing
	}
	pmi := math.Log(cooc) - math.Log(w.totalCooc) - math.Log(math.Pow(c.totalCooc, contextDistributionSmoothing)) + math.Log(cdsTotal)
	return pmi
}

func (w *Word) LogCoocDirect(c *Word, cooc float64) float64 {
	if cooc < 1 {
		cooc = 1
	}
	return math.Log(cooc)
}

// The following are required by Go's sorting facility to sort the vocabulary by freq.
type ByFreq []*Word

func (a ByFreq) Len() int           { return len(a) }
func (a ByFreq) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByFreq) Less(i, j int) bool { return a[i].freq >= a[j].freq }

// Creates corpus scanner using custom scanWords
func createScanner(reader io.Reader) *bufio.Scanner {
	var s = bufio.NewScanner(bufio.NewReader(reader))
	s.Split(scanWords)
	return s
}

// Custom scanWords that converts newlines and periods to ctxbreakbytes
func scanWords(data []byte, atEOF bool) (advance int, token []byte, err error) {
	// Skip leading spaces.
	start := 0
	for width := 0; start < len(data); start += width {
		var r rune
		r, width = utf8.DecodeRune(data[start:])
		if r == '\n' || r == '.' {
			return start + width, ctxbreakbytes, nil
		}
		if !isSpace(r) {
			break
		}
	}
	// Scan until space, marking end of word.
	for width, i := 0, start; i < len(data); i += width {
		var r rune
		r, width = utf8.DecodeRune(data[i:])
		if isSpace(r) {
			if r == '\n' || r == '.' {
				width = 0
			}
			return i + width, data[start:i], nil
		}
	}
	// If we're at EOF, we have a final, non-empty, non-terminated word. Return it.
	if atEOF && len(data) > start {
		return len(data), data[start:], nil
	}
	// Request more data.
	return start, nil, nil
}

// Identical to stdlib but allows treating periods as whitespace
func isSpace(r rune) bool {
	if r <= '\u00FF' {
		// Obvious ASCII ones: \t through \r plus space. Plus two Latin-1 oddballs.
		if periodIsWhitespace && r == '.' {
			return true
		}
		switch r {
		case ' ', '\t', '\n', '\v', '\f', '\r':
			return true
		case '\u0085', '\u00A0':
			return true
		}
		return false
	}
	// High-valued ones.
	if '\u2000' <= r && r <= '\u200a' {
		return true
	}
	switch r {
	case '\u1680', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000':
		return true
	}
	return false
}

// Calculates the probability of subsampling word w given threshold t.
func (w *Word) SubsampleP(t float64, corpusSize uint64) float64 {
	if t == 0 {
		return 0
	}
	subsampleP := 1 - math.Sqrt(t/(float64(w.freq)/float64(corpusSize)))
	if subsampleP < 0 {
		subsampleP = 0
	}
	return subsampleP
}

// Complement of subsampling probability.
func (w *Word) KeepP(t float64, corpusSize uint64) float64 {
	return 1 - w.SubsampleP(t, corpusSize)
}

// Negative sampling interface, implemented by unigram and uniform samplers.
type Sampler interface {
	Sample(r *rand.Rand) *Word
}

// Unigram sampling with context distribution smoothing, implements Sampler interface.
// Ported from word2vec.
type UnigramDist struct {
	vocab []*Word
	table []int
}

func NewUnigramDist(vocab []*Word, table_size int, power float64) *UnigramDist {
	var train_words_pow float64
	table := make([]int, table_size)
	vocab_size := len(vocab)
	for i := 0; i < vocab_size; i++ {
		w := vocab[i]
		train_words_pow += math.Pow(float64(w.freq), power)
	}
	var i int
	d1 := math.Pow(float64(vocab[i].freq), power) / train_words_pow
	for a := 0; a < table_size; a++ {
		table[a] = i
		if float64(a)/float64(table_size) > d1 {
			i++
			d1 += math.Pow(float64(vocab[i].freq), power) / train_words_pow
		}
		if i >= vocab_size {
			i = vocab_size - 1
		}
	}
	return &UnigramDist{vocab, table}
}

func (d *UnigramDist) Sample(r *rand.Rand) *Word {
	i := r.Intn(len(d.table))
	w := d.vocab[d.table[i]]
	return w
}

// Performs SGD. When using in-memory 1) uses CoocStorage to find
// # coocs between words w and c 2) uses Sampler to get negative samples.
// When using external memory 1) CoocStorage is ignored and directCooc is used
// instead (it's coming from a file, not in-memory) 2) Sampler is ignored.
// deltaVec (gradients for each word vector) is supplied by each thread to reduce GC.
func learn(mapw, mapc *Word, coocStorage *CoocStorage, noiseSampler Sampler, r *rand.Rand, noiseSamples int, deltaVec []float64, alpha float64, directCooc float64) float64 {
	for j := uint32(0); j < dim; j++ {
		deltaVec[j] = 0
	}
	err := float64(0)
	deltaVecBias := float64(0)
	w := mapw.i
	for k := 0; k < noiseSamples+1; k++ {
		if k > 0 {
			mapc = noiseSampler.Sample(r)
			if mapw == mapc {
				continue
			}
			for mapc == ctxbreakw {
				mapc = noiseSampler.Sample(r)
			}
		}
		c := mapc.i
		dot := float64(0)
		for j := uint32(0); j < dim; j++ {
			dot += mVec[w*dim+j] * mCtx[c*dim+j]
		}
		g := float64(0)
		if processing == PROC_EXTERNAL_MEMORY_SGD {
			g = dot - directCooc
		} else {
			cooc := coocStorage.GetCooc(mapw, mapc)
			g = dot - cooc
		}
		if useBias {
			g += bVec[w] + bCtx[c]
		}
		err += 0.5 * g * g
		g *= alpha
		for j := uint32(0); j < dim; j++ {
			mVecG := g * mCtx[c*dim+j]
			deltaVec[j] += mVecG
			mCtxG := g * mVec[w*dim+j]
			mCtxGAdj := mCtxG
			if adagrad {
				mCtxGAdj /= math.Sqrt(mCtxGrad[c*dim+j])
				mCtxGrad[c*dim+j] += mCtxG * mCtxG
			}
			mCtx[c*dim+j] -= mCtxGAdj
			if math.IsNaN(deltaVec[j]) || math.IsNaN(mCtx[c*dim+j]) {
				panic("nan")
			}
		}
		if useBias {
			bVecG := g
			deltaVecBias += bVecG
			bCtxG := g
			bCtxGAdj := bCtxG
			if adagrad {
				bCtxGAdj /= math.Sqrt(bCtxGrad[c])
				bCtxGrad[c] += bCtxG * bCtxG
			}
			bCtx[c] -= bCtxGAdj
		}
		if math.IsNaN(deltaVecBias) || math.IsNaN(bCtx[c]) {
			panic("nan")
		}
	}
	for j := uint32(0); j < dim; j++ {
		mVecGAdj := deltaVec[j]
		if adagrad {
			mVecGAdj /= math.Sqrt(mVecGrad[w*dim+j])
			mVecGrad[w*dim+j] += deltaVec[j] * deltaVec[j]
		}
		mVec[w*dim+j] -= mVecGAdj
	}
	if useBias {
		bVecGAdj := deltaVecBias
		if adagrad {
			bVecGAdj /= math.Sqrt(bVecGrad[w])
			bVecGrad[w] += deltaVecBias * deltaVecBias
		}
		bVec[w] -= bVecGAdj
	}
	return err
}

// RingBuffer was used in a previous version of LexVec which supported continuous streams of
// text without sentence breaks, not limiting sentences to a maximum length. Iis now being used
// as a simple vector of length MAX_SENTENCE_LENGTH. It's still in the code in case we re-add
// support for continuous streams.
type RingBuffer struct {
	arr              []RingBufferNode
	c, n, start, end int
}

type RingBufferNode struct {
	v interface{}
}

// call with array
func NewRingBuffer(c int) *RingBuffer {
	return &RingBuffer{make([]RingBufferNode, c), c, 0, 0, 0}
}

func (r *RingBuffer) Len() int {
	return r.n
}

func (r *RingBuffer) Clear() {
	r.start, r.end, r.n = 0, 0, 0
}

// push to end
func (r *RingBuffer) Push(v interface{}) {
	if r.n == r.c {
		r.Pop()
	}
	r.arr[r.end].v = v
	r.end = (r.end + 1) % r.c
	r.n++
}

// removes first in
func (r *RingBuffer) Pop() interface{} {
	if r.n == 0 {
		return nil
	}
	ret := r.arr[r.start].v
	r.start = (r.start + 1) % r.c
	r.n--
	return ret
}

// get
func (r *RingBuffer) Get(i int) interface{} {
	if i >= r.n {
		return nil
	}
	return r.arr[(r.start+i)%r.c].v
}

// HERE BE DRAGONS! This never-ending main routine is way overdue for refactoring.
// I started with a prototype which grew and grew, and with deadlines upon me,
// refactoring was postponed. Things were already grim, but when I added the
// positional contexts and external memory implementation (with yet another
// deadline looming!), it became quite a mess. I will get around to refactoring this
// thing some time soon, but until then I've added (tons of) comments to help you (and me!)
// make sense of it.
//
// This immense main routine is divided into 4 STEPS:
//
// STEP 1: Construct the vocabulary.
// STEP 2: Construct the cooccurence matrix.
// STEP 3: Train the vectors via SGD.
// STEP 4: Output the trained vectors.
//
// If you would like to factorize a matrix other than PPMI, PMI, or LOGCOC,
// implement the interface used the *Direct functions above and
// grep for ADD YOUR CODE to find where to add the call to your function
// for transforming the cooc matrix.
//
// Good luck!

func main() {
	randng = rand.New(rand.NewSource(1))
	var corpusPath = flag.String("corpus", "", "path to corpus")
	var vocabPath = flag.String("savevocab", "", "path where to output vocab")
	var readVocabPath = flag.String("readvocab", "", "path where to read vocab")
	var initialAlpha = flag.Float64("alpha", 0.025, "learning rate")
	var subsample = flag.Float64("subsample", 1e-5, "subsampling threshold")
	flag.Float64Var(&contextDistributionSmoothing, "cds", 0.75, "context distribution smoothing")
	var dimRaw = flag.Int("dim", 300, "number of dimensions of word vectors")
	var iterations = flag.Int("iterations", 5, "how many times to process corpus")
	flag.IntVar(&window, "window", 2, "symmetric window of (window, word, window)")
	var postWindow = flag.Int("postwindow", 0, "post symmetric window of (window, word, window); if 0 it is set -window")
	var minFreq = flag.Int("minfreq", 100, "remove from vocab words that occur less that this number of times")
	var maxVocab = flag.Int("maxvocab", 0, "max vocab size, 0 for no limit")
	var decayAlpha = flag.Bool("decay", true, "decaying learning rate")
	var noise = flag.Int("negative", 5, "number of negative samples")
	var sgNoise = flag.Bool("minibatch", false, "negative sampling per w,c pair rather than per window")
	var unigramPower = flag.Float64("unigrampow", 0.75, "raise unigram dist to this power")
	var weightedWindow = flag.Bool("weightwindow", false, "use randomized window size from uniform(1, window)")
	var postWeightedWindow = flag.Bool("postweightwindow", false, "use randomized postwindow size from uniform(1, window)")
	var model = flag.Int("model", 1, "0 = output W, C; 1 = output W; 2 = output W + C")
	var numThreads = flag.Int("threads", 12, "number of threads to use")
	flag.BoolVar(&externalMemory, "externalmemory", false, "use external memory")
	flag.BoolVar(&useBias, "bias", false, "use bias")
	flag.BoolVar(&adagrad, "adagrad", false, "use adagrad")
	flag.Float64Var(&postSubsample, "postsubsample", 0, "subsampling during SGD; if 0 it is set to -subsample")
	flag.StringVar(&matrix, "matrix", PPMI_MATRIX, "which matrix to factor ("+PPMI_MATRIX+","+PMI_MATRIX+","+LOG_COOC_MATRIX+","+COOC_MATRIX+") default = "+PPMI_MATRIX)
	flag.IntVar(&verbose, "verbose", DEBUG, "verboseness (0 = errors only, 1 = info, 2 = debug) default = 1")
	var printCooc = flag.Bool("printcooc", false, "print coocs for external memory use")
	var coocTotalsPath = flag.String("cooctotalspath", "", "path to cooc totals for each word when using external memory")
	var coocPath = flag.String("coocpath", "", "path to coocs when using external memory")
	var mi = flag.Bool("mi", false, "use MI (multiple instance) rather than SI  when using external memory")
	flag.BoolVar(&positionalContexts, "pos", true, "use positional contexts")
	var vectorOutputPath = flag.String("output", "", "where to save vectors")
	flag.BoolVar(&periodIsWhitespace, "periodiswhitespace", false, "treat period as whitespace")

	flag.Usage = func() {
		fmt.Printf("Usage: lexvec [options]\nOptions:\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// if we don't want different window sizes for cooc matrix construction and SGD,
	// set them to be the same.
	if *postWindow == 0 {
		*postWindow = window
	}

	// if we don't want different subsampling thresholds for
	// cooc matrix construction and SGD, set them to be the same.
	if postSubsample == 0 {
		postSubsample = *subsample
	}

	// cast int to uint32 (flag pkg expects so we convert it here)
	dim = uint32(*dimRaw)

	var err error

	// these streams are writing/reading cooc counts
	var coocStream *os.File
	var coocStreamFileSize int64

	// Use flags to identify if we are processing 1) in-memory 2) external-memory
	// pre-processing (count and print coocs) 3) external-memory SGD. Not using -processing
	// as a flag directly to support legacy scripts.
	if *printCooc && !externalMemory {
		logit("FATAL ERROR: -printcooc must be used with -externalmemory", true, ERROR)
		os.Exit(1)
	}
	processing = PROC_IN_MEMORY
	if *printCooc {
		processing = PROC_EXTERNAL_MEMORY_PRE
	} else if externalMemory {
		processing = PROC_EXTERNAL_MEMORY_SGD
	}

	// if we are using external memory, need to open the
	// the cooc files for reading or writing
	if processing == PROC_EXTERNAL_MEMORY_PRE || processing == PROC_EXTERNAL_MEMORY_SGD {
		if len(*coocPath) == 0 || len(*coocTotalsPath) == 0 {
			logit("FATAL ERROR: coocpath and cooctotalspath are required arguments", true, ERROR)
			os.Exit(1)
		}
		if processing == PROC_EXTERNAL_MEMORY_PRE {
			coocStream, err = os.Create(*coocPath)
			check(err)
		} else {
			coocStream, err = os.Open(*coocPath)
			check(err)
			coocStat, err := os.Stat(*coocPath)
			check(err)
			coocStreamFileSize = coocStat.Size()
		}

	}

	// corpus stream used to create scanners
	var corpus *os.File
	var corpusFileSize int64

	// if we are printing coocs or not using external memory, we need access to the
	// corpus to count the coocs.
	if processing == PROC_EXTERNAL_MEMORY_PRE || processing == PROC_IN_MEMORY {
		if len(*corpusPath) == 0 {
			logit("FATAL ERROR: corpus is a required argument", true, ERROR)
			os.Exit(1)
		}
		corpus, err = os.Open(*corpusPath)
		check(err)
		corpusStat, err := os.Stat(*corpusPath)
		check(err)
		corpusFileSize = corpusStat.Size()
	}

	// If we're not printing coocs we are doing SGD and so
	// need somewhere to output the vectors.
	if processing != PROC_EXTERNAL_MEMORY_PRE {
		if len(*vectorOutputPath) == 0 {
			logit("FATAL ERROR: output is a required argument", true, ERROR)
			os.Exit(1)
		}
	}

	//////////////////////////////////////
	// STEP 1: Vocabulary construction  //
	//////////////////////////////////////

	// Create vocabulary map for mapping tokens to Word and inverse vocabulary map
	// (should convert to list since indexed by contiguous int!)
	vocab := make(map[string]*Word)
	var vocabSize uint32
	var vocabList []*Word
	iVocab := make(map[uint32]*Word)

	// Set the context vocabulary to be the same vocabulary.
	ctxVocab := vocab
	var ctxVocabSize uint32
	var ctxVocabList []*Word
	iCtxVocab := iVocab

	// If we are using positional contexts, need to create separate vocabulary.
	if positionalContexts {
		ctxVocab = make(map[string]*Word)
		iCtxVocab = make(map[uint32]*Word)
	}

	// If we supply a vocabulary file, read vocabulary from file
	if len(*readVocabPath) > 0 {
		logit("reading vocab", true, INFO)
		vocabFile, err := os.Open(*readVocabPath)
		check(err)
		s := createScanner(vocabFile)
		for s.Scan() {
			w := s.Text()
			s.Scan()
			freq, err := strconv.ParseUint(s.Text(), 10, 64)
			check(err)
			vocab[w] = &Word{w, 0, freq, 0}
			s.Scan() // kill context-break
		}

		// If we are using positional contexts, need to read the positional counts
		// from a separate file "readVocabPath + CONTEXT_SUFFIX".
		if positionalContexts {
			var i uint32
			logit("reading context vocab", true, INFO)
			vocabFile, err := os.Open(*readVocabPath + CONTEXT_SUFFIX)
			check(err)
			s := bufio.NewScanner(vocabFile)
			s.Split(bufio.ScanLines)
			for s.Scan() {
				parts := strings.Split(s.Text(), " ")
				w := parts[0]
				coocsString := parts[1]
				coocs, err := strconv.ParseFloat(coocsString, 64)
				check(err)
				mapw := &Word{w, i, uint64(coocs), 0}
				ctxVocab[w] = mapw
				iCtxVocab[i] = mapw
				ctxVocabList = append(ctxVocabList, mapw)
				i++
			}
			ctxVocabSize = i
		}
	} else {
		// Didn't supply a vocabulary file. Will go over corpus to find words and their counts.
		logit("build vocab", true, INFO)
		s := createScanner(corpus)
		for s.Scan() {
			if vocabSize%1000 == 0 {
				logit(fmt.Sprintf("%d\r", vocabSize), false, DEBUG)
			}
			tok := s.Text()
			_, ok := vocab[tok]
			if !ok {
				vocab[tok] = &Word{tok, 0, 0, 0}
				vocabSize++
			}
			vocab[tok].freq += 1
		}
	}

	// If vocab doesn't contain specail CTXBREAK word, add it to vocab.
	if _, ok := vocab[CTXBREAK]; !ok {
		vocab[CTXBREAK] = &Word{CTXBREAK, 0, 0, 0}
	}

	// Remember the CTXBREAK Word (used all over the place).
	ctxbreakw = vocab[CTXBREAK]
	for _, v := range vocab {
		vocabList = append(vocabList, v)
	}

	// Now sort the vocab by frequency and discard words if their frequency
	// is below minFreq or cap vocab if its size exceeds maxVocab.
	var i = uint32(0)
	var newVocabList []*Word
	sort.Sort(ByFreq(vocabList))
	for _, w := range vocabList {
		rawCorpusSize += w.freq
		if w.freq < uint64(*minFreq) || (*maxVocab > 0 && i >= uint32(*maxVocab)) {
			delete(vocab, w.w)
			continue
		}
		w.i = i
		iVocab[i] = w
		i++
		newVocabList = append(newVocabList, w)
		corpusSize += w.freq
	}

	vocabList = newVocabList

	// If we are not using positional contexts, copy the sorted list of vocab words.
	if !positionalContexts {
		ctxVocabList = vocabList
	} else if len(ctxVocabList) == 0 {
		// In this case we are using positional contexts and did not supply a vocab file
		// (had we supplied one ctxVocabList would already exist),
		// so we need to create the corresponding positional Word for each Word
		// in the vocabulary. Their initial frequency is zero since we did not look at
		// cooc windows when constructing the vocab. This will be done later
		// in the program when counting coocs.
		var i uint32
		logit("creating positional vocab words", true, INFO)
		for _, w := range vocabList {
			if w == ctxbreakw {
				continue
			}
			for j := -window; j <= window; j++ {
				if j == 0 {
					continue
				}
				posW := w.posW(j)
				w := &Word{posW, i, 0, 0}
				ctxVocab[posW] = w
				iCtxVocab[i] = w
				i++
				ctxVocabList = append(ctxVocabList, w)
			}
		}
	}

	vocabSize = uint32(len(vocabList))
	ctxVocabSize = uint32(len(ctxVocabList))

	logit(fmt.Sprintf("vocab size: %d\ncontext vocab size: %d\ncorpus size: %d\nraw corpus size (only valid when constructing vocab): %d", vocabSize, ctxVocabSize, corpusSize, rawCorpusSize), true, INFO)

	/////////////////////////////////////
	// STEP 2: Co-occurrence counting  //
	/////////////////////////////////////

	// If we printing coocs or not using external memory, we need to slide a window over the
	// corpus and count word coocurrences.

	logit("initializing cooc storage", true, DEBUG)

	// This stores the coocs for (w,c) pairs. This particular implementation is
	// backed by a list of dictionaries (one for each word).
	coocStorage := &CoocStorage{NewDictMatrixStorage(0., vocabSize, ctxVocabSize)}

	// Only count coocs if printing coocs or not using external memory. Otherwise coocs will
	// be read as input by external memory implementation directly from coocStream.
	if processing == PROC_EXTERNAL_MEMORY_PRE || processing == PROC_IN_MEMORY {
		logit("identify coocurrence", true, INFO)

		// Rewing corpus in case it was ready when constructing vocab.
		corpus.Seek(0, 0)
		s := createScanner(corpus)

		coocurrenceCounter := uint64(0)

		// This buffer will store the current sentence.
		buf := NewRingBuffer(MAX_SENTENCE_LENGTH)

		// If we are printing coocs, create a writer on top of coocStream.
		var coocStreamWriter *bufio.Writer
		if processing == PROC_EXTERNAL_MEMORY_PRE {
			coocStreamWriter = bufio.NewWriter(coocStream)
		}

		// Iterate over every token in the corpus.
		for s.Scan() {
			coocurrenceCounter++
			if coocurrenceCounter%1000 == 0 {
				logit(fmt.Sprintf("%d\r", coocurrenceCounter), false, DEBUG)
			}
			wordInStream := s.Text()
			mapw, ok := vocab[wordInStream]
			// If word is not in vocabulary, skip it.
			if !ok {
				continue
			}

			// Run subsampling, skipping word with probability subsampleP.
			if mapw != ctxbreakw && *subsample > 0 {
				subsampleP := mapw.SubsampleP(*subsample, corpusSize)
				if subsampleP > 0 {
					bernoulliTrial := randng.Float64() // uniform dist 0.0 - 1.0
					if bernoulliTrial <= subsampleP {
						continue
					}
				}
			}

			// If we hit a context break (newline or period) or the buffer is full,
			// run window over the buffer and count coocs.
			if mapw == ctxbreakw || buf.Len() == MAX_SENTENCE_LENGTH {
				// j is the position of target word within sliding window.
				for j := 0; j < buf.Len(); j++ {
					target := buf.Get(j).(*Word)
					win := window
					// If we are using weighted window like word2vec, uniformly sample window size.
					if *weightedWindow {
						win = 1 + randng.Intn(window)
					}
					start := max(0, j-win)
					end := min(buf.Len(), j+win+1)
					// Iterate over each word in window except for target word.
					for i := start; i < end; i++ {
						if i == j {
							continue
						}

						mapc := buf.Get(i).(*Word)
						// If using positional contexts, convert token to token_i
						if positionalContexts {
							posW := mapc.posW(i - j)
							mapc, _ = ctxVocab[posW]
						}

						inc := float64(1)
						// Increment cooc count
						target.totalCooc += inc

						// If using positional contexts, increment the totalCooc of mapc
						// since the cooc is not symmetric.
						if positionalContexts {
							mapc.totalCooc += inc
						}

						// Print the cooc instance if doing pre-processing for external memory
						// else using in-memory increment the stored cooc count.
						if processing == PROC_EXTERNAL_MEMORY_PRE {
							coocStreamWriter.WriteString(target.w)
							coocStreamWriter.WriteString(" ")
							coocStreamWriter.WriteString(mapc.w)
							coocStreamWriter.WriteString("\n")
						} else {
							coocStorage.AddCooc(target, mapc, inc)
						}
					}
				}
				// Clear the buffer so we can process the next sentence.
				buf.Clear()
			}

			// If the current target word is not a context break, push it into the buffer.
			if mapw != ctxbreakw {
				buf.Push(mapw)
			}
		}

		// If using positional contexts, need to set the frequency of context word to
		// totalCooc since this was not done when constructing the vocab and is
		// required for negative sampling.
		if positionalContexts {
			for _, w := range ctxVocabList {
				w.freq = uint64(w.totalCooc)
			}
		}

		// If pre-processing for external memory, need to write negative samples
		// to the coocStream.
		if processing == PROC_EXTERNAL_MEMORY_PRE {
			logit("creating vocab sampling distribution", true, INFO)
			// Need to sort ctxVocabList since cooc counting will have set frequency
			// if using positional contexts.
			sort.Sort(ByFreq(ctxVocabList))
			noiseSampler := NewUnigramDist(ctxVocabList, 1e8, *unigramPower)

			logit("adding negative samples", true, INFO)
			coocurrenceCounter = 0

			// The following loop is equivalent to running over the corpus and
			// drawing negative samples.
			for _, mapw := range vocabList {
				// Don't draw negative samples for context breaks.
				if mapw == ctxbreakw {
					continue
				}

				// Expected number of samples for a Word
				numSamples := uint64(math.Ceil(float64(mapw.freq) * mapw.KeepP(*subsample, corpusSize) * float64(*noise)))
				for i := uint64(0); i < numSamples; i++ {
					coocurrenceCounter++
					if coocurrenceCounter%1000 == 0 {
						logit(fmt.Sprintf("%d\r", coocurrenceCounter), false, DEBUG)
					}
					mapc := noiseSampler.Sample(randng)
					if mapw == mapc {
						continue
					}
					for mapc == ctxbreakw {
						mapc = noiseSampler.Sample(randng)

					}
					coocStreamWriter.WriteString(mapw.w)
					coocStreamWriter.WriteString(" ")
					coocStreamWriter.WriteString(mapc.w)
					// Note the trailing asterisk! Used to identify negative samples in
					// coocStream
					coocStreamWriter.WriteString(" *")
					coocStreamWriter.WriteString("\n")
				}
			}
			coocStreamWriter.Flush()
		}
	}

	// Now we can save the vocab if we specified a save path. Note this could not
	// be done in STEP 1 when using positional contexts because cooc counting was
	// necessary to get the frequency of context words.
	if len(*vocabPath) > 0 {
		logit("saving vocab", true, INFO)
		vocabOutput, err := os.Create(*vocabPath)
		if err != nil {
			panic("unable to open vocab path")
		}
		for _, w := range vocabList {
			fmt.Fprintf(vocabOutput, "%s %d\n", w.w, w.freq)
		}
		vocabOutput.Close()
		if positionalContexts {
			vocabOutput, err := os.Create(*vocabPath + CONTEXT_SUFFIX)
			if err != nil {
				panic("unable to open context vocab path")
			}
			for _, w := range ctxVocabList {
				fmt.Fprintf(vocabOutput, "%s %f\n", w.w, w.totalCooc)
			}
			vocabOutput.Close()
		}
	}

	// If pre-processing for external memory, write cooctotals to disk and exit.
	if processing == PROC_EXTERNAL_MEMORY_PRE {
		coocTotalsOutput, err := os.Create(*coocTotalsPath)
		check(err)
		logit("writing cooc totals", true, INFO)
		for _, w := range vocabList {
			fmt.Fprintf(coocTotalsOutput, "%s %f\n", w.w, w.totalCooc)
		}
		coocTotalsOutput.Close()

		logit("done", true, INFO)
		os.Exit(0)
	}

	// The following paragraph is a duplicate of the code
	// for printing negative samples above. Refactor me please!
	// START COPYANDPASTE
	// Need to sort ctxVocabList since cooc counting will have set frequency
	// if using positional contexts.
	sort.Sort(ByFreq(ctxVocabList))
	noiseSampler := NewUnigramDist(ctxVocabList, 1e8, *unigramPower)
	// END COPYANDPASTE

	// When using external memory, need to read cooc totals so that association measures
	// can be calculated. If not using external memory, these are already available in the
	// Word values since the program did not exit after cooc counting.
	if processing == PROC_EXTERNAL_MEMORY_SGD {
		logit("reading cooc totals", true, INFO)
		coocTotalsStream, err := os.Open(*coocTotalsPath)
		check(err)
		s := bufio.NewScanner(coocTotalsStream)
		s.Split(bufio.ScanLines)
		for s.Scan() {
			parts := strings.Split(s.Text(), " ")
			w := parts[0]
			coocsString := parts[1]
			coocs, err := strconv.ParseFloat(coocsString, 64)
			check(err)
			vocab[w].totalCooc = coocs
		}
		if positionalContexts {
			for _, w := range ctxVocabList {
				w.totalCooc = float64(w.freq)
			}
		}
		coocTotalsStream.Close()
	}

	// cdsTotal will be used for context distribution smoothing the the negative
	// sampling distribution
	for _, w := range ctxVocabList {
		cdsTotal += math.Pow(w.totalCooc, contextDistributionSmoothing)
	}
	logit(fmt.Sprintf("cds total: %f", cdsTotal), true, INFO)

	// If using in-memory, we transform the cooc matrix
	// using the specified association measure. ADD YOUR CODE if
	// you want to use a different measure.
	if processing == PROC_IN_MEMORY {
		logit("calculating "+matrix+" matrix", true, INFO)
		coocStorage.Transform(func(row, col uint32, v float64) float64 {
			w := vocabList[row]
			c := iCtxVocab[col]
			switch matrix {
			case PPMI_MATRIX:
				return w.PpmiDirect(c, v)
			case PMI_MATRIX:
				return w.PmiDirect(c, v)
			case LOG_COOC_MATRIX:
				return w.LogCoocDirect(c, v)
				// case COOC_MATRIX not needed as is exactly p
			}
			return v
		})
	}

	///////////////////
	// STEP 3: SGD   //
	///////////////////

	// Create the word and context vectors. These are the values we
	// are ultimately interested in.
	mVec = make([]float64, vocabSize*dim)
	mCtx = make([]float64, ctxVocabSize*dim)

	// Create bias vectors if are using bias in SGD.
	bVec = make([]float64, vocabSize)
	bCtx = make([]float64, ctxVocabSize)

	// Adagrad requires one parameter per model parameter.
	if adagrad {
		mVecGrad = make([]float64, vocabSize*dim)
		bVecGrad = make([]float64, vocabSize)
		mCtxGrad = make([]float64, ctxVocabSize*dim)
		bCtxGrad = make([]float64, ctxVocabSize)
	}

	// Initialize word vectors and biases to Uniform(-.5,.5) / dim
	// and Adagrad params to 1.
	logit("create vectors", true, INFO)
	for j := uint32(0); j < vocabSize; j++ {
		for k := uint32(0); k < dim; k++ {
			mVec[j*dim+k] = (randng.Float64() - 0.5) / float64(dim)
			if adagrad {
				mVecGrad[j*dim+k] = 1.0
			}
		}
		bVec[j] = (randng.Float64() - 0.5) / float64(dim)
		if adagrad {
			bVecGrad[j] = 1.0
		}
	}

	// Repeat for context vectors.
	for j := uint32(0); j < ctxVocabSize; j++ {
		for k := uint32(0); k < dim; k++ {
			mCtx[j*dim+k] = (randng.Float64() - 0.5) / float64(dim)
			if adagrad {
				mCtxGrad[j*dim+k] = 1.0
			}
		}
		bCtx[j] = (randng.Float64() - 0.5) / float64(dim)
		if adagrad {
			bCtxGrad[j] = 1.0
		}
	}

	logit("running lexvec", true, INFO)

	// Create counters per thread to avoid synchronization.
	processed := make([]uint64, *numThreads)
	bytesRead := make([]uint64, *numThreads)
	for j := 0; j < *numThreads; j++ {
		processed[j] = 0
		bytesRead[j] = 0
	}

	// This is used to wait on SGD threads until all are complete.
	var wg sync.WaitGroup

	// For reporting error.
	avgError := float64(0)
	avgErrorNum := uint64(0)

	// Current learning-rate. Will be linearly-decayed as processing progresses.
	alpha := *initialAlpha

	// Start SGD threads.
	for threadId := 0; threadId < *numThreads; threadId++ {
		wg.Add(1)
		go func(threadId int) {
			// NOW WE ARE IN AN SGD THREAD

			// Each thread needs its own random number generator. This is only used
			// by in-memory implementation. No randomness in external memory.
			randn := rand.New(rand.NewSource(int64(threadId)))

			// Used by learn() to store gradients. Shared accross all calls in same
			// thread to avoid GC.
			deltaVec := make([]float64, dim)

			// Here there's a big fork:
			// - if we are using external memory we will run SGD over
			//       lines from coocStream.
			// - else we are using in-memory, will slide window over corpus,
			//       use subsampling, and run SGD negative sampling

			if processing == PROC_EXTERNAL_MEMORY_SGD {
				// EXTERNAL MEMORY LexVec

				// This thread needs its own handle to coocPath
				coocStream, err := os.Open(*coocPath)
				check(err)

				// Determine thread's segment within file.
				coocStreamOffsetStart := (coocStreamFileSize / int64(*numThreads)) * int64(threadId)
				coocStreamOffsetEnd := (coocStreamFileSize / int64(*numThreads)) * int64(threadId+1)
				if coocStreamOffsetEnd >= coocStreamFileSize {
					coocStreamOffsetEnd = coocStreamFileSize
				}

				// Run iterations (epochs) over segment
				for iter := 0; iter < *iterations; iter++ {
					_, err := coocStream.Seek(coocStreamOffsetStart, 0)
					lastCurPos := uint64(coocStreamOffsetStart)
					check(err)

					s := bufio.NewScanner(coocStream)
					// Note that unlike corpus scanner here we use bufio.ScanLines to scan
					// lines and not tokens
					s.Split(bufio.ScanLines)

					// Consume first line to make sure we're not using a
					// partial sentence. If empty, exit thread.
					if !s.Scan() {
						logit("got nothing, exiting", true, INFO)
						wg.Done()
						return
					}

					// Run over each line in segment.
					for s.Scan() {
						// This funky Seek call is to get current position within file.
						curPos, aerr := coocStream.Seek(0, 1)
						check(aerr)

						// Check to see if overstepping segment. If so done with iteration.
						if curPos > coocStreamOffsetEnd {
							break
						}

						// Update progress counters and decay learning rate if necessary.
						bytesRead[threadId] += uint64(curPos) - lastCurPos
						lastCurPos = uint64(curPos)
						processed[threadId]++
						if processed[threadId]%10000 == 0 {
							if *decayAlpha && !adagrad {
								var totalBytesRead uint64
								for j := 0; j < *numThreads; j++ {
									totalBytesRead += bytesRead[j]
								}
								alpha = *initialAlpha * (float64(1) - (float64(totalBytesRead) / (float64(coocStreamFileSize) * float64(*iterations))))
								if alpha < *initialAlpha*0.0001 {
									alpha = *initialAlpha * 0.0001
								}
							}
						}

						// Split a line "w c coocs"
						parts := strings.Split(s.Text(), " ")
						w := parts[0]
						mapw, ok1 := vocab[w]
						c := parts[1]
						mapc, ok2 := ctxVocab[c]
						pText := parts[2]
						p, nok3 := strconv.ParseFloat(pText, 64)
						if !ok1 || !ok2 || nok3 != nil {
							panic("problem parsing")
						}

						learningIterations := uint64(math.Ceil(p))

						// Was line negative sample? "w c count *"
						isNoise := len(parts) > 3 && parts[3] == "*"
						// Last check: line is not a negative example if it ends in a number:
						// "w c count * cooc". This means a negative sample was drawn
						// but the (w,c) had cooccurred.
						if isNoise && len(parts) == 5 {
							// negative sample with non-zero ppmi, cooc is stored in last part
							isNoise = false
							p, nok3 = strconv.ParseFloat(parts[4], 64)
							if nok3 != nil {
								panic("problem parsing2")
							}
						}

						// Transform cooc into association measure. ADD YOUR CODE here
						// if you want to use a different measure.
						y := float64(0)
						if !isNoise {
							switch matrix {
							case PPMI_MATRIX:
								y = mapw.PpmiDirect(mapc, p)
							case PMI_MATRIX:
								y = mapw.PmiDirect(mapc, p)
							case LOG_COOC_MATRIX:
								y = mapw.LogCoocDirect(mapc, p)
							case COOC_MATRIX:
								y = p
							}
						}

						// If we're running in MI mode, run cooc count iterations of SGD
						// for the pair. Otherwise run only 1 (rest of iterations are spread
						// throughout coocStream).
						if !*mi {
							learningIterations = 1
						}

						// Run SGD. Note coocStorage and noiseSampler are not passed
						// as they are not used in external memory (coocs are coming from
						// file, and negative samples were drawn when building cooc file).
						for i := uint64(0); i < learningIterations; i++ {
							err := learn(mapw, mapc, nil, nil, nil, 0, deltaVec, alpha, y)
							avgError += err
							avgErrorNum++
						}
					}

					// Report loss at the end of iteration, only on thread 0.
					if threadId == 0 {
						avgError /= float64(avgErrorNum)
						logit(fmt.Sprintf("iteration %d MSE = %f", iter+1, avgError), true, INFO)
						avgError = 0
						avgErrorNum = 0
					}
				}
			} else {
				// IN-MEMORY LexVec

				// Each thread needs handle to corpus
				corpus, err := os.Open(*corpusPath)
				check(err)

				// Determine thread's segment within file.
				corpusOffsetStart := (corpusFileSize / int64(*numThreads)) * int64(threadId)
				corpusOffsetEnd := (corpusFileSize / int64(*numThreads)) * int64(threadId+1)
				if corpusOffsetEnd >= corpusFileSize {
					corpusOffsetEnd = corpusFileSize
				}

				// This buffer will store the current sentence.
				buf := NewRingBuffer(MAX_SENTENCE_LENGTH)

				// Run iterations (epochs) over segment
				for iter := 0; iter < *iterations; iter++ {
					_, err := corpus.Seek(corpusOffsetStart, 0)
					check(err)

					s := createScanner(corpus)

					// Consume first line to make sure we're not using
					// a partial sentence. If empty, exit thread.
					if !s.Scan() {
						logit("got nothing, exiting", true, INFO)
						wg.Done()
						return
					}

					// Clear the buffer before starting a new iteration
					buf.Clear()

					// The following loop is exactly the same as was used to construct the
					// the cooccurrence matrix, but instead of increment a cell (w,c) in
					// the matrix it runs SGD on (w,c).

					// Iterate over every token in the corpus within thread's segment.
					for s.Scan() {
						// This funky Seek call is to get current position within file.
						curPos, err := corpus.Seek(0, 1)
						check(err)

						// Check to see if overstepping segment. If so done with iteration.
						if curPos > corpusOffsetEnd {
							break
						}

						// Update progress counters and decay learning rate if necessary.
						processed[threadId]++
						if processed[threadId]%10000 == 0 {
							if *decayAlpha && !adagrad {
								var totalProcessed uint64
								for j := 0; j < *numThreads; j++ {
									totalProcessed += processed[j]
								}
								alpha = *initialAlpha * (float64(1) - (float64(totalProcessed) / (float64(rawCorpusSize) * float64(*iterations))))
								if alpha < *initialAlpha*0.0001 {
									alpha = *initialAlpha * 0.0001
								}
							}
						}

						wordInStream := s.Text()
						mapw, ok := vocab[wordInStream]
						// If word is not in vocabulary, skip it.
						if !ok {
							continue
						}

						// Run subsampling, skipping word with probability subsampleP.
						if mapw != ctxbreakw && postSubsample > 0 {
							subsampleP := mapw.SubsampleP(postSubsample, corpusSize)
							if subsampleP > 0 {
								bernoulliTrial := randn.Float64() // uniform dist 0.0 - 1.0
								if bernoulliTrial <= subsampleP {
									continue
								}
							}
						}

						// If we hit a context break (newline or period) or the buffer is full,
						// run window over the buffer and run SGD.
						if mapw == ctxbreakw || buf.Len() == MAX_SENTENCE_LENGTH {
							// j is the position of target word within sliding window.
							for j := 0; j < buf.Len(); j++ {
								target := buf.Get(j).(*Word)
								win := *postWindow
								// If we are using weighted window like word2vec, uniformly sample window size.
								if *postWeightedWindow {
									win = 1 + randn.Intn(*postWindow)
								}
								start := max(0, j-win)
								end := min(buf.Len(), j+win+1)
								// Iterate over each word in window except for target word.
								for i := start; i < end; i++ {
									if i == j {
										continue
									}

									mapc := buf.Get(i).(*Word)
									// If using positional contexts, convert token to token_i
									if positionalContexts {
										posW := mapc.posW(i - j)
										mapc, _ = ctxVocab[posW]
									}

									// If we are using sgnoise, draw negative samples for
									// the current (w,c) pair. This means that for a given
									// target 2*win*noise negative samples will be drawn.
									wNoise := 0
									if *sgNoise {
										wNoise = *noise
									}

									// Run SGD.
									err := learn(target, mapc, coocStorage, noiseSampler, randn, wNoise, deltaVec, alpha, 0)
									// Acccount loss.
									avgError += err
									avgErrorNum++
								}

								// If we are not using sgnoise, draw negative samples for
								// target. This has the same effect as polluting the context
								// window with sampled words.
								if !*sgNoise {
									for i := 0; i < *noise; i++ {
										// Draw negative sample.
										mapc := noiseSampler.Sample(randn)

										// Skip if sample is target word.
										if target == mapc {
											continue
										}

										// Draw until sample is not context break.
										for mapc == ctxbreakw {
											mapc = noiseSampler.Sample(randn)
										}

										// Run SGD. Note that learn() will not draw any
										// negative samples as we are calling it with a
										// (w,c) where c is the negative sample.
										err := learn(target, mapc, coocStorage, noiseSampler, randn, 0, deltaVec, alpha, 0)
										// Account loss.
										avgError += err
										avgErrorNum++
									}
								}
							}
							// Clear the buffer at the end of each sentence.
							buf.Clear()
						}

						// If the current target word is not a context break,
						// push it into the buffer.
						if mapw != ctxbreakw {
							buf.Push(mapw)
						}
					}

					// Report loss at the end of iteration, only on thread 0.
					if threadId == 0 {
						avgError /= float64(avgErrorNum)
						logit(fmt.Sprintf("iteration %d MSE = %f", iter+1, avgError), true, INFO)
						avgError = 0
						avgErrorNum = 0
					}
				}
			}

			// Signal that this thread is finished.
			wg.Done()
		}(threadId)
	}

	// This channel will signal that progress reporting thread should quit.
	quit := make(chan bool)

	// Progress reporting thread.
	// This whole progress reporting should be moved into one of the SGD threads!
	go func() {
		time.Sleep(time.Second)
		var previousProcessed uint64
		previousTime := time.Now()
		for {
			select {
			case <-quit:
				return
			default:
				var totalProcessed uint64
				for j := 0; j < *numThreads; j++ {
					totalProcessed += processed[j]
				}
				speed := float64(totalProcessed-previousProcessed) / float64(*numThreads) / float64(1000) / time.Since(previousTime).Seconds()
				previousProcessed = totalProcessed
				previousTime = time.Now()
				logit(fmt.Sprintf("%d alpha %f speed %.1fk words/thread/s\r", totalProcessed, alpha, speed), false, DEBUG)
				time.Sleep(time.Second)
			}
		}
	}()

	// Wait for SGD threads to finish.
	wg.Wait()

	// Stop progress reporting thread.
	quit <- true

	//////////////////////////////////////
	// STEP 4: Output trained vectors   //
	//////////////////////////////////////

	logit("outputting vectors", true, INFO)
	vectorOutput, err := os.Create(*vectorOutputPath)
	check(err)
	outputStream := bufio.NewWriter(vectorOutput)

	// Write "vocabsize dim" as first line of vector file.
	outputStream.WriteString(fmt.Sprintf("%d %d\n", len(vocabList), dim))

	// Write each word vector.
	for _, w := range vocabList {
		outputStream.WriteString(w.w)
		for j := uint32(0); j < dim; j++ {
			// j'th component of vector
			v := mVec[w.i*dim+j]

			// Model 2 means we add the context vector to the word vector.
			if *model == 2 {
				// If we're not using positional contexts it's simple addition.
				if !positionalContexts {
					v += mCtx[w.i*dim+j]
				} else {
					// We need to sum all of the positional context vectors (one for each
					// position).
					for k := -window; k <= window; k++ {
						if k == 0 {
							continue
						}
						posC := w.posW(k)
						c, _ := ctxVocab[posC]
						v += mCtx[c.i*dim+j]
					}
				}
			}

			fmt.Fprintf(outputStream, " %f", v)
		}
		outputStream.WriteString("\n")
	}
	outputStream.Flush()

	// Model 0 means we also output the context vectors.
	if *model == 0 {
		// Use the same path as the word vectors but add CONTEXT_SUFFIX.
		ctxOutput, err := os.Create(*vectorOutputPath + CONTEXT_SUFFIX)
		check(err)

		ctxOutputStream := bufio.NewWriter(ctxOutput)

		// Write "vocabsize dim" as first line of vector file.
		fmt.Fprintf(ctxOutputStream, "%d %d\n", len(ctxVocabList), dim)
		for _, w := range ctxVocabList {
			ctxOutputStream.WriteString(w.w)
			for j := uint32(0); j < dim; j++ {
				// Write j'th compoment
				fmt.Fprintf(ctxOutputStream, " %f", mCtx[w.i*dim+j])
			}
			ctxOutputStream.WriteString("\n")
		}
		ctxOutputStream.Flush()
	}
	logit("finished!", true, INFO)

	///////////
	// DONE! //
	///////////
}
