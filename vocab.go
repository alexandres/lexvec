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
	"hash/fnv"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	unigramTableSize = 1e8
)

type word struct {
	w            string
	idx          idxUint
	freq         countUint
	totalCooc    countUint
	subwords     []idxUint
	logTotalCooc real
}

// Adds position within window to word.
// ex: i walked the dog -> target = the -> i_-2 walked_-1 the dog_1
func (w *word) posW(pos int) string {
	return fmt.Sprintf("%s_%d", w.w, pos)
}

// ByFreq allows sort.Sort to sort words by freq
type ByFreq []*word

func (a ByFreq) Len() int           { return len(a) }
func (a ByFreq) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByFreq) Less(i, j int) bool { return a[i].freq >= a[j].freq }

type countCallback func(w string, cnt countUint)

func readCounts(s *bufio.Scanner, callback countCallback) {
	for s.Scan() {
		w := s.Text()
		s.Scan()
		freq, err := strconv.ParseUint(s.Text(), 10, 32)
		check(err)
		callback(w, countUint(freq))
		s.Scan()
	}
}

func readVocabFile(path string) ([]*word, map[string]*word) {
	vocabFile, err := os.Open(path)
	check(err)
	s := createScanner(vocabFile)
	var vocabList []*word
	vocab := make(map[string]*word)
	readCounts(s, func(w string, freq countUint) {
		newWord := &word{w, idxUint(len(vocabList)), freq, 0, nil, 0}
		vocabList = append(vocabList, newWord)
		vocab[w] = newWord
	})
	return vocabList, vocab
}

func readVocab() {
	logln(infoLogLevel, "reading vocab")
	vocabList, vocab = readVocabFile(vocabPath)
	logln(infoLogLevel, "reading context vocab")
	ctxVocabList, ctxVocab = readVocabFile(vocabPath + contextPathSuffix)

	vocabSize = idxUint(len(vocabList))
	ctxVocabSize = idxUint(len(ctxVocabList))

	// Remember the CTXBREAK Word (used all over the place).
	var ok bool
	ctxbreakw, ok = vocab[ctxBreakToken]
	if !ok {
		logln(errorLogLevel, "ctxbreak "+ctxBreakToken+" not in vocab")
	}

	for _, w := range vocabList {
		corpusSize += uint64(w.freq)
	}

	logln(infoLogLevel, "vocab size: %d\ncontext vocab size: %d\ncorpus size: %d\nraw corpus size (only valid when constructing vocab): %d", vocabSize, ctxVocabSize, corpusSize, rawCorpusSize)

	logln(infoLogLevel, "creating vocab sampling distribution")
	noiseSampler = newUnigramDist(ctxVocabList, unigramTableSize, unigramPower)

}

func buildVocab() {
	// Didn't supply a vocabulary file. Will go over corpus to find words and their counts.
	tmpVocab := make(map[string]*word)
	logln(infoLogLevel, "build vocab")
	corpus := openCorpus()
	defer corpus.Close()
	s := createScanner(corpus)
	pp := newProgressPrinter(defaultProgressInterval)
	for s.Scan() {
		pp.inc()
		tok := s.Text()
		w, ok := tmpVocab[tok]
		if !ok {
			w = &word{tok, idxUint(len(vocabList)), 0, 0, nil, 0}
			vocabList = append(vocabList, w)
			tmpVocab[tok] = w
		}
		w.freq++
		checkCountIncOverflow(w.freq)

		rawCorpusSize++
	}

	// Now sort the vocab by frequency and discard words if their frequency
	// is below minFreq or cap vocab if its size exceeds maxVocab.
	sort.Sort(ByFreq(vocabList))
	var cut idxUint
	for ; cut < idxUint(len(vocabList)) && vocabList[cut].freq >= minFreq; cut++ {
		corpusSize += uint64(vocabList[cut].freq)
	}
	if maxVocab > 0 && maxVocab < cut {
		cut = maxVocab
	}
	vocabList = vocabList[:cut]
	// reindex and build definitive vocab
	vocab = make(map[string]*word)
	for i, w := range vocabList {
		w.idx = idxUint(i)
		vocab[w.w] = w
	}

	// If vocabList doesn't contain specail CTXBREAK word, add it to vocab.
	inVocab := false
	for _, w := range vocabList {
		if w.w == ctxBreakToken {
			inVocab = true
			break
		}
	}
	if !inVocab {
		vocab[ctxBreakToken] = &word{ctxBreakToken, idxUint(len(vocabList)), 0, 0, nil, 0}
		vocabList = append(vocabList, vocab[ctxBreakToken])
	}
	// Remember the CTXBREAK Word (used all over the place).
	ctxbreakw = vocab[ctxBreakToken]

	// build context vocab
	ctxVocab = make(map[string]*word)
	if positionalContexts {
		logln(infoLogLevel, "creating positional vocab words")
		for _, w := range vocabList {
			if w == ctxbreakw {
				continue
			}
			for j := -window; j <= window; j++ {
				if j == 0 {
					continue
				}
				posW := w.posW(j)
				w := &word{posW, idxUint(len(ctxVocabList)), 0, 0, nil, 0}
				ctxVocab[posW] = w
				ctxVocabList = append(ctxVocabList, w)
			}
		}
	} else if len(ctxVocabList) == 0 {
		// If we are not using positional contexts, copy the sorted list of vocab words.
		for _, w := range vocabList {
			c := &word{w.w, w.idx, 0, 0, nil, 0}
			ctxVocabList = append(ctxVocabList, c)
			ctxVocab[c.w] = c
		}
	}

	// get subsampled corpus freq for contexts. needed for accurate negative sampling
	logln(infoLogLevel, "getting ctx freq")
	corpus.Seek(0, 0)
	s = createScanner(corpus)
	pp = newProgressPrinter(defaultProgressInterval)
	windower(s, randng, false, func(w, c *word, pos int) bool {
		pp.inc()
		c.freq++
		checkCountIncOverflow(c.freq)
		return true
	})
	// reindex ctx vocab
	sort.Sort(ByFreq(ctxVocabList))
	for i, w := range ctxVocabList {
		w.idx = idxUint(i)
	}
}

func saveVocabFile(path string, vocabList []*word) {
	if path == "" {
		logln(errorLogLevel, "no vocab path given")
	}
	vocabOutput, err := os.Create(path)
	check(err)
	defer vocabOutput.Close()
	for _, w := range vocabList {
		fmt.Fprintf(vocabOutput, "%s %d\n", w.w, w.freq)
	}
}

func saveVocab() {
	logln(infoLogLevel, "saving vocab")
	saveVocabFile(vocabPath, vocabList)
	logln(infoLogLevel, "saving context vocab")
	saveVocabFile(vocabPath+contextPathSuffix, ctxVocabList)
}

func buildSubwords() {
	if subwordMinN < 1 || subwordMinN > subwordMaxN {
		logln(errorLogLevel, "minn must be greater than 0 and less or equal to maxn")
	}
	for _, w := range vocabList {
		for _, subword := range computeSubwords(w.w, subwordMinN, subwordMaxN) {
			w.subwords = append(w.subwords, subwordIdx(subword, vocabSize, subwordMatrixRows-vocabSize))
		}
	}
}

func computeSubwords(unwrappedw string, minn, maxn int) (subwords []string) {
	w := fmt.Sprintf("<%s>", unwrappedw)
	if len(w) < minn {
		return
	}
	for i := 0; i <= len(w)-minn; i++ {
		for l := minn; l < len(w) && l <= maxn && i+l <= len(w); l++ {
			subwords = append(subwords, w[i:i+l])
		}
	}
	return
}

func processSubwords() {
	subwordMatrixRows = vocabSize

	// each vocab word has a unique vector
	for i, w := range vocabList {
		w.subwords = append(w.subwords, idxUint(i))
	}

	if subwordMinN < 1 && len(subwordPath) == 0 {
		return
	}

	var buckets idxUint = 2000000 // same as fasttext
	subwordMatrixRows += buckets

	if subwordMinN > 0 {
		logln(infoLogLevel, "building subword information")
		buildSubwords()
	} else if len(subwordPath) > 0 {
		logln(infoLogLevel, "reading subword information")
		subwordFile, err := os.Open(subwordPath)
		check(err)
		defer subwordFile.Close()
		s := bufio.NewScanner(subwordFile)
		s.Split(bufio.ScanLines)
		var foundSubwords idxUint
		for s.Scan() {
			parts := strings.Split(s.Text(), " ")
			if len(parts) < 2 {
				logln(errorLogLevel, "bad subword line: %s", s.Text())
			}
			w, ok := vocab[parts[0]]
			if !ok {
				continue
			}
			foundSubwords++
			wrappedWord := fmt.Sprintf("<%s>", parts[0])
			for _, sw := range parts[1:] {
				if sw == wrappedWord {
					continue // word already has own vector
				}
				w.subwords = append(w.subwords, subwordIdx(sw, vocabSize, buckets))
			}
		}
		if foundSubwords != vocabSize {
			logln(errorLogLevel, "(intersection subword list and vocab) %d != (vocab size) %d. Make sure subword information is generated from vocab list output by LexVec.", foundSubwords, vocabSize)
		}
		logln(infoLogLevel, "found %d subwords", foundSubwords)
	}
}

func subwordIdx(sw string, vocabSize, buckets idxUint) idxUint {
	h := fnv.New32()
	_, err := h.Write([]byte(sw))
	check(err)
	hash := h.Sum32() % buckets
	return vocabSize + hash
}
