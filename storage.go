/*
 * Copyright (c) 2017 Salle, Alexandre <atsalle@inf.ufrgs.br>
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

type CoocStorage struct {
	storage MatrixStorage
}

type TransformFunc func(row, col uint32, v float64) float64

type MatrixStorage interface {
	Get(row, col uint32) float64
	Set(row, col uint32, v float64)
	Transform(f TransformFunc)
}

func (coocStorage *CoocStorage) GetCooc(w, c *Word) float64 {
	return coocStorage.storage.Get(w.i, c.i)
}

func (coocStorage *CoocStorage) SetCooc(w, c *Word, v float64) {
	coocStorage.storage.Set(w.i, c.i, v)
}

func (coocStorage *CoocStorage) AddCooc(w, c *Word, v float64) {
	coocStorage.SetCooc(w, c, coocStorage.GetCooc(w, c)+v)
}

func (coocStorage *CoocStorage) IncCooc(w, c *Word) {
	coocStorage.AddCooc(w, c, 1)
}

func (coocStorage *CoocStorage) Transform(f TransformFunc) {
	coocStorage.storage.Transform(f)
}

type DictMatrixStorage struct {
	initialValue float64
	mem          []map[uint32]float64
}

func NewDictMatrixStorage(initialValue float64, rows, cols uint32) MatrixStorage {
	var mem []map[uint32]float64
	for i := uint32(0); i < rows; i++ {
		mem = append(mem, make(map[uint32]float64))
	}
	return &DictMatrixStorage{initialValue, mem}
}

func (m *DictMatrixStorage) Get(row, col uint32) float64 {
	v, ok := m.mem[row][col]
	if !ok {
		return m.initialValue
	}
	return v
}

func (m *DictMatrixStorage) Set(row, col uint32, v float64) {
	m.mem[row][col] = v
}

func (m *DictMatrixStorage) Transform(f TransformFunc) {
	for i, d := range m.mem {
		for j, v := range d {
			m.mem[i][j] = f(uint32(i), uint32(j), v)
		}
	}
}
