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
	"errors"
	"fmt"
	"os"
)

// Helper for aborting on error.
func check(e error) {
	if e != nil {
		logln(errorLogLevel, "panic: %v", e)
	}
}

func checkCountIncOverflow(v countUint) {
	if v == 0 {
		panic(errors.New("overflow in countUint, change type countUint = uint32 to countUint = uint64"))
	}
}

func log(level int, msg string, args ...interface{}) {
	doLog(level, msg, false, args...)
}

func logln(level int, msg string, args ...interface{}) {
	doLog(level, msg, true, args...)
}

func doLog(level int, msg string, lineBreak bool, args ...interface{}) {
	if verbose < level {
		return
	}
	if lineBreak {
		fmt.Fprintf(os.Stderr, "\n")
	}
	fmt.Fprintf(os.Stderr, msg, args...)
	if lineBreak {
		fmt.Fprintf(os.Stderr, "\n")
	}
	os.Stderr.Sync()
	if level == errorLogLevel {
		os.Exit(1)
	}
}

type progressPrinter struct {
	n   uint64
	mod uint64
}

func newProgressPrinter(mod uint64) *progressPrinter {
	return &progressPrinter{0, mod}
}

func (p *progressPrinter) inc() {
	p.n++
	if p.n%p.mod == 0 {
		log(infoLogLevel, "\r%dK", p.n/1000)
	}
}
