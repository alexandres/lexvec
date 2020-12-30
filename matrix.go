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

type matrix interface {
	get(row, col idxUint) countUint
	set(row, col idxUint, v countUint)
}

type dictMatrix struct {
	initialValue countUint
	mem          []map[idxUint]countUint
}

func newDictMatrix(initialValue countUint, rows, cols idxUint) *dictMatrix {
	var mem []map[idxUint]countUint
	for i := idxUint(0); i < rows; i++ {
		mem = append(mem, make(map[idxUint]countUint))
	}
	return &dictMatrix{initialValue, mem}
}

func (m *dictMatrix) get(row, col idxUint) countUint {
	v, ok := m.mem[row][col]
	if !ok {
		return m.initialValue
	}
	return v
}

func (m *dictMatrix) set(row, col idxUint, v countUint) {
	m.mem[row][col] = v
}
