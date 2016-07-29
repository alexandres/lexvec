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

import "flag"
import "io"
import "os"
import "bufio"
import "strings"
import "strconv"
import "sync"
import "time"

import "math"
import "math/rand"
import "fmt"
import "sort"
import "unicode/utf8"

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

var verbose int
var dim, corpusSize uint64
var mVec, mCtx, bVec, bCtx, mVecGrad, mCtxGrad, bVecGrad, bCtxGrad []float64
var contextDistributionSmoothing, cdsTotal, postSubsample float64
var useBias, adagrad, externalMemory, positionalContexts, periodIsWhitespace bool
var randng *rand.Rand
var matrix string
var window int
var ctxbreakw *Word

var ctxbreakbytes []byte

func check(e error) {
	if e != nil {
		panic(e)
	}
}

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
	i         uint64
	freq      uint64
	cooc      map[uint64]float64
	totalCooc float64
}

func (w *Word) Ppmi(c *Word) float64 {
	cooc, ok := w.cooc[c.i]
	if !ok {
		return 0
	}
	return w.PpmiDirect(c, cooc)
}

func (w *Word) PpmiDirect(c *Word, cooc float64) float64 {
	ppmi := math.Log(cooc) - math.Log(w.totalCooc) - math.Log(math.Pow(c.totalCooc, contextDistributionSmoothing)) + math.Log(cdsTotal)
	if ppmi < 0 {
		return 0
	}
	return ppmi
}

func (w *Word) Pmi(c *Word) float64 {
	cooc, ok := w.cooc[c.i]
	if !ok {
		cooc = 0
	}
	return w.PmiDirect(c, cooc)
}

func (w *Word) PmiDirect(c *Word, cooc float64) float64 {
	if cooc < 1 {
		cooc = 1 // smoothing
	}
	pmi := math.Log(cooc) - math.Log(w.totalCooc) - math.Log(math.Pow(c.totalCooc, contextDistributionSmoothing)) + math.Log(cdsTotal)
	return pmi
}

func (w *Word) LogCooc(c *Word) float64 {
	cooc, ok := w.cooc[c.i]
	if !ok {
		return 0
	}
	return w.LogCoocDirect(c, cooc)
}

func (w *Word) LogCoocDirect(c *Word, cooc float64) float64 {
	if cooc < 1 {
		cooc = 1
	}
	return math.Log(cooc)
}

func (w *Word) posW(pos int) string {
	return fmt.Sprintf("%s_%d", w.w, pos)
}

type ByFreq []*Word

func (a ByFreq) Len() int           { return len(a) }
func (a ByFreq) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByFreq) Less(i, j int) bool { return a[i].freq >= a[j].freq }

func ShuffleVocab(a []*Word) {
	n := len(a)
	for i := n - 1; i > 0; i-- {
		j := randng.Intn(i + 1)
		a[i], a[j] = a[j], a[i]
	}
}

func createScanner(reader io.Reader) *bufio.Scanner {
	var s = bufio.NewScanner(bufio.NewReader(reader))
	s.Split(scanWords)
	return s
}

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

type Sampler interface {
	Sample(r *rand.Rand) *Word
}

type UnigramDist struct {
	vocab []*Word
	table []int
}

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

func (w *Word) KeepP(t float64, corpusSize uint64) float64 {
	return 1 - w.SubsampleP(t, corpusSize)
}

func NewUnigramDist(vocab []*Word, table_size int, power float64) *UnigramDist {
	// ported from w2v implementation
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

func learn(mapw, mapc *Word, noiseSampler Sampler, r *rand.Rand, noiseSamples int, deltaVec []float64, alpha float64, directCooc float64) float64 {
	for j := uint64(0); j < uint64(dim); j++ {
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
		for j := uint64(0); j < uint64(dim); j++ {
			dot += mVec[w*dim+j] * mCtx[c*dim+j]
		}
		g := float64(0)
		if externalMemory {
			g = dot - directCooc
			if useBias {
				g += bVec[w] + bCtx[c]
			}
		} else {
			cooc, ok := mapw.cooc[mapc.i]
			if !ok {
				cooc = 0
			}
			g = dot - cooc
			if useBias {
				g += bVec[w] + bCtx[c]
			}
		}
		err += 0.5 * g * g
		g *= alpha
		for j := uint64(0); j < uint64(dim); j++ {
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
	for j := uint64(0); j < uint64(dim); j++ {
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
	if *postWindow == 0 {
		*postWindow = window
	}
	if postSubsample == 0 {
		postSubsample = *subsample
	}
	dim = uint64(*dimRaw)
	var err error
	var coocStream *os.File
	var coocStreamFileSize int64
	if *printCooc || externalMemory {
		if len(*coocPath) == 0 || len(*coocTotalsPath) == 0 {
			logit("FATAL ERROR: coocpath and cooctotalspath are required arguments", true, ERROR)
			os.Exit(1)
		}
		if *printCooc {
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
	var corpus *os.File
	var corpusFileSize int64
	if *printCooc || !externalMemory {
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
	if !externalMemory || !*printCooc {
		if len(*vectorOutputPath) == 0 {
			logit("FATAL ERROR: output is a required argument", true, ERROR)
			os.Exit(1)
		}
	}
	vocab := make(map[string]*Word)
	var vocabSize uint64
	var vocabList []*Word
	iVocab := make(map[uint64]*Word)
	ctxVocab := vocab
	var ctxVocabSize uint64
	var ctxVocabList []*Word
	iCtxVocab := iVocab
	if positionalContexts {
		ctxVocab = make(map[string]*Word)
		iCtxVocab = make(map[uint64]*Word)
	}
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
			vocab[w] = &Word{w, 0, freq, make(map[uint64]float64), 0}
			s.Scan() // kill context-break
		}
		if positionalContexts {
			var i uint64
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
				mapw := &Word{w, i, uint64(coocs), make(map[uint64]float64), coocs}
				ctxVocab[w] = mapw
				iCtxVocab[i] = mapw
				ctxVocabList = append(ctxVocabList, mapw)
				i++
			}
			ctxVocabSize = i
		}
	} else {
		logit("build vocab", true, INFO)
		s := createScanner(corpus)
		for s.Scan() {
			if vocabSize%1000 == 0 {
				logit(fmt.Sprintf("%d\r", vocabSize), false, DEBUG)
			}
			tok := s.Text()
			_, ok := vocab[tok]
			if !ok {
				vocab[tok] = &Word{tok, 0, 0, make(map[uint64]float64), 0}
				vocabSize++
			}
			vocab[tok].freq += 1
		}
	}
	var i = uint64(0)
	var newVocabList []*Word
	if _, ok := vocab[CTXBREAK]; !ok {
		vocab[CTXBREAK] = &Word{CTXBREAK, 0, 0, make(map[uint64]float64), 0}
	}
	ctxbreakw = vocab[CTXBREAK]
	for _, v := range vocab {
		vocabList = append(vocabList, v)
	}
	sort.Sort(ByFreq(vocabList))
	for _, w := range vocabList {
		if w.freq < uint64(*minFreq) {
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
	if !positionalContexts {
		ctxVocabList = vocabList
	} else if len(ctxVocabList) == 0 {
		var i uint64
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
				w := &Word{posW, i, 0, make(map[uint64]float64), 0}
				ctxVocab[posW] = w
				iCtxVocab[i] = w
				i++
				ctxVocabList = append(ctxVocabList, w)
			}
		}
	}
	vocabSize = uint64(len(vocabList))
	ctxVocabSize = uint64(len(ctxVocabList))
	logit(fmt.Sprintf("vocab size: %d\ncontext vocab size: %d\ncorpus size: %d", vocabSize, ctxVocabSize, corpusSize), true, INFO)
	var noiseSampler Sampler
	if !externalMemory || *printCooc {
		logit("identify coocurrence", true, INFO)
		corpus.Seek(0, 0)
		s := createScanner(corpus)
		coocurrenceCounter := uint64(0)

		// RingBuffer was used in a previous version of LexVec which supported continuous streams of
		// text without sentence breaks, not limiting sentences to a maximum length. Iis now being used
		// as a simple vector of length MAX_SENTENCE_LENGTH. It's still in the code in case we re-add
		// support for continuous streams.
		buf := NewRingBuffer(MAX_SENTENCE_LENGTH)

		var coocStreamWriter *bufio.Writer
		if *printCooc {
			coocStreamWriter = bufio.NewWriter(coocStream)
		}
		for s.Scan() {
			coocurrenceCounter++
			if coocurrenceCounter%1000 == 0 {
				logit(fmt.Sprintf("%d\r", coocurrenceCounter), false, DEBUG)
			}
			wordInStream := s.Text()
			mapw, ok := vocab[wordInStream]
			if !ok {
				continue
			}
			if mapw != ctxbreakw && *subsample > 0 {
				subsampleP := mapw.SubsampleP(*subsample, corpusSize)
				if subsampleP > 0 {
					bernoulliTrial := randng.Float64() // uniform dist 0.0 - 1.0
					if bernoulliTrial <= subsampleP {
						continue
					}
				}
			}
			if mapw == ctxbreakw || buf.Len() == MAX_SENTENCE_LENGTH {
				// process, clear buf, add token
				for j := 0; j < buf.Len(); j++ {
					target := buf.Get(j).(*Word)
					win := window
					if *weightedWindow {
						win = 1 + randng.Intn(window)
					}
					start := max(0, j-win)
					end := min(buf.Len(), j+win+1)
					for i := start; i < end; i++ {
						if i == j {
							continue
						}
						mapc := buf.Get(i).(*Word)
						if positionalContexts {
							posW := mapc.posW(i - j)
							mapc, _ = ctxVocab[posW]
						}
						inc := float64(1)
						target.totalCooc += inc
						if positionalContexts {
							mapc.totalCooc += inc
						}
						if *printCooc {
							coocStreamWriter.WriteString(target.w)
							coocStreamWriter.WriteString(" ")
							coocStreamWriter.WriteString(mapc.w)
							coocStreamWriter.WriteString("\n")
						} else {
							cooc, ok := target.cooc[mapc.i]
							if !ok {
								cooc = 0
							}
							target.cooc[mapc.i] = cooc + inc
						}
					}
				}
				buf.Clear()
			}
			if mapw != ctxbreakw {
				buf.Push(mapw)
			}
		}
		if positionalContexts {
			for _, w := range ctxVocabList {
				w.freq = uint64(w.totalCooc)
			}
		}
		if *printCooc {
			logit("creating vocab sampling distribution", true, INFO)
			noiseSampler = NewUnigramDist(ctxVocabList, 1e8, *unigramPower)
			// now add the negative samples
			logit("adding negative samples", true, INFO)
			coocurrenceCounter = 0
			for _, mapw := range vocabList {
				if mapw == ctxbreakw {
					continue
				}
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
					coocStreamWriter.WriteString(" *")
					coocStreamWriter.WriteString("\n")
				}
			}
			coocStreamWriter.Flush()
		}
	}
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
	if externalMemory && *printCooc && len(*coocTotalsPath) > 0 {
		coocTotalsOutput, err := os.Create(*coocTotalsPath)
		check(err)
		logit("writing cooc totals", true, INFO)
		for _, w := range vocabList {
			fmt.Fprintf(coocTotalsOutput, "%s %f\n", w.w, w.totalCooc)
		}
		coocTotalsOutput.Close()
	}
	if *printCooc {
		logit("done", true, INFO)
		os.Exit(0)
	}
	sort.Sort(ByFreq(ctxVocabList))
	noiseSampler = NewUnigramDist(ctxVocabList, 1e8, *unigramPower)
	if externalMemory && !*printCooc {
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
		coocTotalsStream.Close()
	}
	for _, w := range ctxVocabList {
		cdsTotal += math.Pow(w.totalCooc, contextDistributionSmoothing)
	}
	logit(fmt.Sprintf("cds total: %f", cdsTotal), true, INFO)
	if !externalMemory {
		logit("calculating "+matrix+" matrix", true, INFO)
		for _, w := range vocabList {
			ppmiTotal := float64(0)
			for c, p := range w.cooc {
				switch matrix {
				case PPMI_MATRIX:
					p = w.Ppmi(iCtxVocab[c])
				case PMI_MATRIX:
					p = w.Pmi(iCtxVocab[c])
				case LOG_COOC_MATRIX:
					p = w.LogCooc(iCtxVocab[c])
					// case COOC_MATRIX not needed as is exactly p
				}
				w.cooc[c] = p
				ppmiTotal += p * p
			}
		}
	}
	mVec = make([]float64, vocabSize*dim)
	mCtx = make([]float64, ctxVocabSize*dim)
	bVec = make([]float64, vocabSize)
	bCtx = make([]float64, ctxVocabSize)
	if adagrad {
		mVecGrad = make([]float64, vocabSize*dim)
		bVecGrad = make([]float64, vocabSize)
		mCtxGrad = make([]float64, ctxVocabSize*dim)
		bCtxGrad = make([]float64, ctxVocabSize)
	}
	logit("create vectors", true, INFO)
	for j := uint64(0); j < vocabSize; j++ {
		for k := uint64(0); k < dim; k++ {
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
	for j := uint64(0); j < ctxVocabSize; j++ {
		for k := uint64(0); k < dim; k++ {
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
	processed := uint64(0)
	bytesRead := uint64(0)
	var wg sync.WaitGroup
	avgError := float64(0)
	avgErrorNum := uint64(0)
	refAlpha := *initialAlpha
	for threadId := 0; threadId < *numThreads; threadId++ {
		wg.Add(1)
		go func(threadId int) {
			alpha := *initialAlpha
			randn := rand.New(rand.NewSource(int64(threadId)))
			deltaVec := make([]float64, dim)
			if externalMemory {
				coocStream, err := os.Open(*coocPath)
				check(err)
				coocStreamOffsetStart := (coocStreamFileSize / int64(*numThreads)) * int64(threadId)
				coocStreamOffsetEnd := (coocStreamFileSize / int64(*numThreads)) * int64(threadId+1)
				if coocStreamOffsetEnd >= coocStreamFileSize {
					coocStreamOffsetEnd = coocStreamFileSize
				}
				for iter := 0; iter < *iterations; iter++ {
					_, err := coocStream.Seek(coocStreamOffsetStart, 0)
					check(err)

					s := bufio.NewScanner(coocStream)
					s.Split(bufio.ScanLines)
					// eat first line to make sure no junk
					if !s.Scan() {
						logit("got nothing, exiting", true, INFO)
						wg.Done()
						return
					}
					for s.Scan() {
						curPos, aerr := coocStream.Seek(0, 1)
						check(aerr)
						if curPos > coocStreamOffsetEnd {
							break
						}
						bytesRead += uint64(len(s.Text()))
						processed++
						if processed%1000 == 0 {
							if *decayAlpha && !adagrad {
								alpha = *initialAlpha * (float64(1) - (float64(bytesRead) / (float64(coocStreamFileSize) * float64(*iterations))))
								if alpha < *initialAlpha*0.0001 {
									alpha = *initialAlpha * 0.0001
								}
							}
							if threadId == 0 {
								refAlpha = alpha
							}
						}
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
						isNoise := len(parts) > 3 && parts[3] == "*"
						learningIterations := uint64(math.Ceil(p))
						if isNoise && len(parts) == 5 {
							// negative sample with non-zero ppmi, cooc is stored in last part
							isNoise = false
							p, nok3 = strconv.ParseFloat(parts[4], 64)
							if nok3 != nil {
								panic("problem parsing2")
							}
						}
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
						if !*mi {
							learningIterations = 1
						}
						for i := uint64(0); i < learningIterations; i++ {
							err := learn(mapw, mapc, noiseSampler, randn, 0, deltaVec, alpha, y)
							avgError += err
							avgErrorNum++
						}
					}
					if threadId == 0 {
						avgError /= float64(avgErrorNum)
						logit(fmt.Sprintf("iteration %d MSE = %f", iter+1, avgError), true, INFO)
						avgError = 0
						avgErrorNum = 0
					}
				}
			} else {
				corpus, err := os.Open(*corpusPath)
				check(err)
				corpusOffsetStart := (corpusFileSize / int64(*numThreads)) * int64(threadId)
				corpusOffsetEnd := (corpusFileSize / int64(*numThreads)) * int64(threadId+1)
				if corpusOffsetEnd >= corpusFileSize {
					corpusOffsetEnd = corpusFileSize
				}
				buf := NewRingBuffer(MAX_SENTENCE_LENGTH)
				for iter := 0; iter < *iterations; iter++ {
					_, err := corpus.Seek(corpusOffsetStart, 0)
					check(err)

					s := createScanner(corpus)
					if !s.Scan() {
						// consume first word in case it's partial
						logit("got nothing, exiting", true, INFO)
						wg.Done()
						return
					}
					buf.Clear()
					for s.Scan() {
						curPos, err := corpus.Seek(0, 1)
						check(err)
						if curPos > corpusOffsetEnd {
							break
						}
						processed++
						if processed%1000 == 0 {
							if *decayAlpha && !adagrad {
								alpha = *initialAlpha * (float64(1) - (float64(processed) / (float64(corpusSize) * float64(*iterations))))
								if alpha < *initialAlpha*0.0001 {
									alpha = *initialAlpha * 0.0001
								}
							}
							if threadId == 0 {
								refAlpha = alpha
							}
						}
						wordInStream := s.Text()
						mapw, ok := vocab[wordInStream]
						if !ok {
							continue
						}
						if mapw != ctxbreakw && postSubsample > 0 {
							subsampleP := mapw.SubsampleP(postSubsample, corpusSize)
							if subsampleP > 0 {
								bernoulliTrial := randn.Float64() // uniform dist 0.0 - 1.0
								if bernoulliTrial <= subsampleP {
									continue
								}
							}
						}
						if mapw == ctxbreakw || buf.Len() == MAX_SENTENCE_LENGTH {
							// process, clear buf, add token
							for j := 0; j < buf.Len(); j++ {
								target := buf.Get(j).(*Word)
								win := *postWindow
								if *postWeightedWindow {
									win = 1 + randn.Intn(*postWindow)
								}
								start := max(0, j-win)
								end := min(buf.Len(), j+win+1)
								for i := start; i < end; i++ {
									if i == j {
										continue
									}
									mapc := buf.Get(i).(*Word)
									if positionalContexts {
										posW := mapc.posW(i - j)
										mapc, _ = ctxVocab[posW]
									}
									wNoise := 0
									if *sgNoise {
										wNoise = *noise
									}
									err := learn(target, mapc, noiseSampler, randn, wNoise, deltaVec, alpha, 0)
									avgError += err
									avgErrorNum++
								}
								if !*sgNoise {
									for i := 0; i < *noise; i++ {
										mapc := noiseSampler.Sample(randn)
										if target == mapc {
											continue
										}
										for mapc == ctxbreakw {
											mapc = noiseSampler.Sample(randn)
										}
										err := learn(target, mapc, noiseSampler, randn, 0, deltaVec, alpha, 0)
										avgError += err
										avgErrorNum++
									}
								}
							}
							buf.Clear()
						}
						if mapw != ctxbreakw {
							buf.Push(mapw)
						}
					}
					if threadId == 0 {
						avgError /= float64(avgErrorNum)
						logit(fmt.Sprintf("iteration %d MSE = %f", iter+1, avgError), true, INFO)
						avgError = 0
						avgErrorNum = 0
					}
				}
			}
			wg.Done()
		}(threadId)
	}
	var done = false
	go func() {
		time.Sleep(time.Second)
		previousProcessed := processed
		previousTime := time.Now()
		for !done {
			speed := float64(processed-previousProcessed) / float64(*numThreads) / float64(1000) / time.Since(previousTime).Seconds()
			previousProcessed = processed
			previousTime = time.Now()
			logit(fmt.Sprintf("%d alpha %f speed %.1fk words/thread/s\r", processed, refAlpha, speed), false, DEBUG)
			time.Sleep(time.Second)
		}
	}()
	wg.Wait()
	done = true
	logit("outputting vectors", true, INFO)
	vectorOutput, err := os.Create(*vectorOutputPath)
	check(err)
	outputStream := bufio.NewWriter(vectorOutput)
	outputStream.WriteString(fmt.Sprintf("%d %d\n", len(vocabList), dim))
	for _, w := range vocabList {
		outputStream.WriteString(w.w)
		for j := uint64(0); j < dim; j++ {
			v := mVec[w.i*dim+j]
			if *model == 2 {
				if !positionalContexts {
					v += mCtx[w.i*dim+j]
				} else {
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
	if *model == 0 {
		ctxOutput, err := os.Create(*vectorOutputPath + CONTEXT_SUFFIX)
		check(err)
		ctxOutputStream := bufio.NewWriter(ctxOutput)
		fmt.Fprintf(ctxOutputStream, "%d %d\n", len(ctxVocabList), dim)
		for _, w := range ctxVocabList {
			ctxOutputStream.WriteString(w.w)
			for j := uint64(0); j < dim; j++ {
				fmt.Fprintf(ctxOutputStream, " %f", mCtx[w.i*dim+j])
			}
			ctxOutputStream.WriteString("\n")
		}
		ctxOutputStream.Flush()
	}
	logit("finished!", true, INFO)
}
