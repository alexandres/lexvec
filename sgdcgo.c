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

#include "sgdcgo.h"	

realPtr mVec, mCtx; 
idxUintPtr wSubwordOffsets, wSubwordIdxs;
idxUint vocabSize, wSubwordIdxsLen, dim;

void sgdStepCgoInit(idxUint _vocabSize, realPtr _mVec, realPtr _mCtx, idxUint _wSubwordIdxsLen, idxUintPtr _wSubwordIdxs, idxUintPtr _wSubwordOffsets, idxUint _dim) {
	vocabSize = _vocabSize;
	mVec = _mVec;
	mCtx = _mCtx;
	wSubwordIdxsLen = _wSubwordIdxsLen;
	wSubwordIdxs = _wSubwordIdxs;
	wSubwordOffsets = _wSubwordOffsets;
	dim = _dim;
}

double sgdStepCgo(idxUint wIdx, idxUint cIdx, real y, realPtr zVec, real alpha) {
	idxUint swStart = wSubwordOffsets[wIdx];
	idxUint swEnd = wIdx < vocabSize - 1 ? wSubwordOffsets[wIdx+1] : wSubwordIdxsLen;
	idxUint wSubwordLen = swEnd - swStart;
	for (idxUint j = 0; j < dim; j++) {
		zVec[j] = 0;
	}
	for (idxUint i = swStart; i < swEnd; i++) {
		for (idxUint j = 0; j < dim; j++) {
			zVec[j] += mVec[wSubwordIdxs[i]*dim+j];
		}
	}
	real dot = 0;
	for (idxUint j = 0; j < dim; j++) {
		zVec[j] /= wSubwordLen;
		dot += zVec[j] * mCtx[cIdx*dim+j];
	}
	real g = dot - y;
	real err = 0.5 * g * g;
	g *= alpha;
	for (idxUint j = 0; j < dim; j++) {
		real mVecG = g * mCtx[cIdx*dim+j] / wSubwordLen;
		real mCtxG = g * zVec[j];
		zVec[j] = mVecG; // reusing zVec to store gradients so loop below can be vectorized
		mCtx[cIdx*dim+j] -= mCtxG;
	}
	for (idxUint i = swStart; i < swEnd; i++) {
		for (idxUint j = 0; j < dim; j++) {		
			mVec[wSubwordIdxs[i]*dim+j] -= zVec[j];
		}
	}
	return err;
}

double sgdStepCgoBatch(idxUintPtr wIdx, idxUintPtr cIdx, realPtr y, realPtr zVec, real alpha, idxUint n) {
	double loss = 0;
	for (idxUint i = 0; i < n; i++) {
		loss += sgdStepCgo(wIdx[i], cIdx[i], y[i], zVec, alpha);
	}
	return loss;
}
