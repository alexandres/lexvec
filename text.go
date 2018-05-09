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
	"io"
	"math/rand"
	"os"
	"unicode/utf8"
)

func createScanner(reader io.Reader) *bufio.Scanner {
	var s = bufio.NewScanner(bufio.NewReader(reader))
	s.Split(scanWords)
	return s
}

func openCorpus() *os.File {
	if len(corpusPath) == 0 {
		logln(errorLogLevel, "FATAL ERROR: corpus is a required argument")
		os.Exit(1)
	}
	corpus, err := os.Open(corpusPath)
	check(err)
	return corpus
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

type windowerCallback func(w, c *word, pos int) bool

func windower(s *bufio.Scanner, randng *rand.Rand, includeTargetOnly bool, callback windowerCallback) {
	var buf []*word

	// Iterate over every token in the corpus.
	for s.Scan() {
		wordInStream := s.Text()
		mapw, ok := vocab[wordInStream]
		// If word is not in vocabulary, skip it.
		if !ok {
			continue
		}

		// Run subsampling, skipping word with probability subsampleP.
		if mapw != ctxbreakw && subsample > 0 {
			subsampleP := mapw.subsampleP(subsample, corpusSize)
			if subsampleP > 0 {
				bernoulliTrial := randng.Float64() // uniform dist 0.0 - 1.0
				if bernoulliTrial <= subsampleP {
					continue
				}
			}
		}

		// If we hit a context break (newline or period) or the buffer is full,
		// run window over the buffer.
		if mapw == ctxbreakw || len(buf) == maxSentenceLen {
			// j is the position of target word within sliding window.
			for j, target := range buf {
				win := window
				// If we are using weighted window like word2vec, uniformly sample window size.
				if weightedWindow {
					win = 1 + randng.Intn(window)
				}

				start := j - win
				if start < 0 {
					start = 0
				}
				end := j + win + 1
				if end > len(buf) {
					end = len(buf)
				}

				// Iterate over each word in window except for target word.
				for i := start; i < end; i++ {
					if i == j {
						continue
					}

					mapc := buf[i]
					// If using positional contexts, convert token to token_i
					pos := i - j
					posW := mapc.w
					if positionalContexts {
						posW = mapc.posW(pos)
					}
					mapc, ok = ctxVocab[posW]
					if !ok {
						logln(errorLogLevel, "ctx word %s not in vocab", posW)
					}

					if !callback(target, mapc, pos) {
						return
					}
				}
				if includeTargetOnly {
					if !callback(target, nil, 0) {
						return
					}
				}
			}
			// Clear the buffer so we can process the next sentence.
			buf = nil
		}

		// If the current target word is not a context break, push it into the buffer.
		if mapw != ctxbreakw {
			buf = append(buf, mapw)
		}
	}
}
