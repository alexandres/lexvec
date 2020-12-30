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

import "math"

var associationMap = map[string]associationMeasureFunc{
	"cpmi":    cPmiAssociationMeasureFunc,
	"npmi":    nPmiAssociationMeasureFunc,
	"nnegpmi": nNegPmiAssociationMeasureFunc,
}

type associationMeasureFunc func(w, c *word, cooc countUint) real

func cPmiAssociationMeasureFunc(w, c *word, cooc countUint) real {
	if cooc == 0 {
		return clipPmi
	}
	pmi := pmiAssociationMeasureFunc(w, c, cooc)
	if pmi < clipPmi {
		return clipPmi
	}
	return pmi
}

func pmiAssociationMeasureFunc(w, c *word, cooc countUint) real {
	return math.Log(real(cooc)) - w.logTotalCooc - c.logTotalCooc + logCdsTotal
}

func nPmiAssociationMeasureFunc(w, c *word, cooc countUint) real {
	if cooc == 0 {
		return -1
	}
	return pmiAssociationMeasureFunc(w, c, cooc) / -(math.Log(real(cooc)) - logCdsTotal)
}

func nNegPmiAssociationMeasureFunc(w, c *word, cooc countUint) real {
	if cooc == 0 {
		return -1
	}
	pmi := pmiAssociationMeasureFunc(w, c, cooc)
	if pmi >= 0 {
		return pmi
	}
	return nPmiAssociationMeasureFunc(w, c, cooc)
}
