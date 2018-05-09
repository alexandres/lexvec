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

// #include "sgdcgo.h"
import "C"

import (
	"unsafe"
)

var wSubwordIdxs, wSubwordOffsets []idxUint

func sgdStepCgoInit() {
	for _, w := range vocabList {
		wSubwordOffsets = append(wSubwordOffsets, idxUint(len(wSubwordIdxs)))
		for _, swIdx := range w.subwords {
			wSubwordIdxs = append(wSubwordIdxs, swIdx)
		}
	}
	C.sgdStepCgoInit(
		C.idxUint(vocabSize),
		C.realPtr(unsafe.Pointer(&mVec[0])),
		C.realPtr(unsafe.Pointer(&mCtx[0])),
		C.idxUint(len(wSubwordIdxs)),
		C.idxUintPtr(unsafe.Pointer(&wSubwordIdxs[0])),
		C.idxUintPtr(unsafe.Pointer(&wSubwordOffsets[0])),
		C.idxUint(dim),
	)
}

// zVec (gradient for word vector) is supplied by each thread to reduce GC.
func sgdStepCgoBatch(wIdx []idxUint, cIdx []idxUint, y []real, zVec []real, alpha real, n idxUint) real {
	return float64(
		C.sgdStepCgoBatch(
			C.idxUintPtr(unsafe.Pointer(&wIdx[0])),
			C.idxUintPtr(unsafe.Pointer(&cIdx[0])),
			C.realPtr(unsafe.Pointer(&y[0])),
			C.realPtr(unsafe.Pointer(&zVec[0])),
			C.real(alpha),
			C.idxUint(n)))
}

// zVec (gradient for word vector) is supplied by each thread to reduce GC.
func sgdStep(mapw, mapc *word, y real, zVec []real, alpha real) real {
	for j := idxUint(0); j < dim; j++ {
		zVec[j] = 0
	}
	c := mapc.idx
	var dot real
	zSize := real(len(mapw.subwords))
	for _, sw := range mapw.subwords {
		for j := idxUint(0); j < dim; j++ {
			zVec[j] += mVec[sw*dim+j]
		}
	}
	for j := idxUint(0); j < dim; j++ {
		zVec[j] /= zSize
		dot += zVec[j] * mCtx[c*dim+j]
	}
	g := dot - y
	err := 0.5 * g * g
	g *= alpha
	for j := idxUint(0); j < dim; j++ {
		mVecG := g * mCtx[c*dim+j] / zSize
		mCtxG := g * zVec[j]
		for _, sw := range mapw.subwords {
			mVec[sw*dim+j] -= mVecG
		}
		mCtx[c*dim+j] -= mCtxG
	}
	return err
}
