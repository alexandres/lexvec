package main

import (
	"reflect"
	"testing"
)

func init() {
	verbose = DEBUG
}

func TestBinaryEncoding(t *testing.T) {
	p := Uint64Pair{5, 6}
	q := CacheBytesPair(PairToCacheKey(p.I, p.J))
	if !reflect.DeepEqual(p, q) {
		t.Error("p", p, "q", q, "not equal")
	}
	f := float64(0.000000002)
	g := CacheBytesToFloat64(Float64ToCacheBytes(f))
	if f != g {
		t.Error(f, g, "not equal")
	}
}

func TestLevelDBKV(t *testing.T) {
	kv, err := NewLevelDBStore("somegarbage")
	defer kv.Cleanup()
	if err != nil {
		t.Error(err)
	}
	p := Uint64Pair{5, 6}
	k := PairToCacheKey(p.I, p.J)
	f := 1.0
	v := Float64ToCacheBytes(f)
	err = kv.Put(k, v)
	if err != nil {
		t.Error(err)
	}
	z, err := kv.Get(k)
	if err != nil {
		t.Error(err)
	}
	g := CacheBytesToFloat64(z)
	if f != g {
		t.Error(f, g, "not equal")
	}
	p2 := Uint64Pair{1, 2}
	k2 := PairToCacheKey(p2.I, p2.J)
	f2 := 2.0
	v2 := Float64ToCacheBytes(f2)
	err = kv.Put(k2, v2)
	if err != nil {
		t.Error(err)
	}
	seen := 0
	err = kv.Iterate(func(tk, tv []byte) {
		seen++
	})
	if seen != 2 {
		t.Error(seen, "not 2")
	}
}

func TestCacheStorage(t *testing.T) {
	kv, err := NewLevelDBStore("somegarbage")
	if err != nil {
		t.Error(err)
	}
	defer kv.Cleanup()
	s := NewCachedMatrixStorage(kv, 0., 10, 20, 0.)
	p := Uint64Pair{0, 0}
	f := 2.
	s.Set(p.I, p.J, f)
	g := s.Get(p.I, p.J)
	if f != g {
		t.Error(f, g, "not equal")
	}
	p2 := Uint64Pair{1, 1}
	g2 := s.Get(p2.I, p2.J)
	if s.initialValue != g2 {
		t.Error(s.initialValue, g2, "not equal")
	}
	total := float64(0)
	s.Transform(func(i, j uint64, x float64) float64 {
		total += x
		return x * 2
	})
	if f != total {
		t.Error(f, total, "not equal")
	}
	newf := f * 2
	g3 := s.Get(p.I, p.J)
	if newf != g3 {
		t.Error(newf, g3)
	}
}
