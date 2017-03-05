package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"unsafe"

	"github.com/syndtr/goleveldb/leveldb"
	leveldberrors "github.com/syndtr/goleveldb/leveldb/errors"
	leveldbopt "github.com/syndtr/goleveldb/leveldb/opt"
)

type CoocStorage struct {
	storage MatrixStorage
}

type TransformFunc func(row, col uint64, v float64) float64

type MatrixStorage interface {
	Get(row, col uint64) float64
	Set(row, col uint64, v float64)
	Transform(f TransformFunc)
	ReadOnly() error
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

func (coocStorage *CoocStorage) ReadOnly() error {
	return coocStorage.storage.ReadOnly()
}

type DictMatrixStorage struct {
	initialValue float64
	mem          map[Uint64Pair]float64
}

func NewDictMatrixStorage(initialValue float64, rows, cols uint64) MatrixStorage {
	mem := make(map[Uint64Pair]float64)
	return &DictMatrixStorage{initialValue, mem}
}

func (m *DictMatrixStorage) Get(row, col uint64) float64 {
	v, ok := m.mem[Uint64Pair{row, col}]
	if !ok {
		return m.initialValue
	}
	return v
}

func (m *DictMatrixStorage) Set(row, col uint64, v float64) {
	m.mem[Uint64Pair{row, col}] = v
}

func (m *DictMatrixStorage) Transform(f TransformFunc) {
	for pair, v := range m.mem {
		m.mem[pair] = f(pair.I, pair.J, v)
	}
}

func (m *DictMatrixStorage) ReadOnly() error {
	return nil
}

type IterateFunc func(key, val []byte)

type KVStore interface {
	Get(key []byte) ([]byte, error)
	Put(key, val []byte) error
	ReopenReadOnly() error
	Iterate(f IterateFunc) error
	Cleanup()
}

type LevelDBStore struct {
	dbPath string
	db     *leveldb.DB
	opts   leveldbopt.Options
}

var ErrKeyNotFound = errors.New("Key not found")

func NewLevelDBStore(dbPath string, cacheSize, blockSize, writeBuffer int) (*LevelDBStore, error) {
	dbPath, err := ioutil.TempDir(dbPath, "coocdb")
	check(err)
	opts := leveldbopt.Options{}
	// opts.BlockCacher = leveldbopt.NoCacher
	opts.NoSync = true
	// opts.WriteBuffer = 1024 * 1024 * 1000 // 1000mb
	opts.Compression = leveldbopt.NoCompression
	opts.WriteBuffer = writeBuffer
	opts.BlockCacheCapacity = cacheSize
	opts.BlockSize = blockSize
	_ = os.RemoveAll(dbPath)
	db, err := leveldb.OpenFile(dbPath, &opts)
	return &LevelDBStore{dbPath, db, opts}, err
}

func (ldb *LevelDBStore) Get(key []byte) ([]byte, error) {
	val, err := ldb.db.Get(key, nil)
	if err != nil {
		if err == leveldberrors.ErrNotFound {
			err = ErrKeyNotFound
		} else {
			panic(err)
		}
	}
	return val, err
}

func (ldb *LevelDBStore) Put(key, val []byte) error {
	return ldb.db.Put(key, val, nil)
}

func (ldb *LevelDBStore) Iterate(f IterateFunc) error {
	iter := ldb.db.NewIterator(nil, nil)
	for iter.Next() {
		f(iter.Key(), iter.Value())
	}
	return nil
}

func (ldb *LevelDBStore) ReopenReadOnly() error {
	logit("reopening readonly", true, DEBUG)
	err := ldb.db.Close()
	if err != nil {
		return err
	}
	ldb.opts.ReadOnly = true
	db, err := leveldb.OpenFile(ldb.dbPath, &ldb.opts)
	ldb.db = db
	return err
}

func (ldb *LevelDBStore) Cleanup() {
	logit("cleaning leveldb", true, DEBUG)
	err := ldb.db.Close()
	check(err)
	err = os.RemoveAll(ldb.dbPath)
	check(err)
}

type CachedMatrixStorage struct {
	initialValue float64
	rows         uint64
	cols         uint64
	cache        MatrixStorage
	kv           KVStore
	cacheSize    uint64
	maxProduct   uint64
}

func NewCachedMatrixStorage(kv KVStore, initialValue float64, rows, cols uint64, cacheSizeinGB float64) *CachedMatrixStorage {
	var t1 float64
	var t2 Uint64Pair
	cellSizeInBytes := (unsafe.Sizeof(t1) + unsafe.Sizeof(t2))
	valuesPerGb := float64(1e9 / cellSizeInBytes)

	// copied from https://github.com/stanfordnlp/GloVe/blob/master/src/cooccur.c
	rlimit := cacheSizeinGB * valuesPerGb
	var n float64
	n = 1e5
	for math.Abs(rlimit-n*(math.Log(n)+0.1544313298)) > 1e-3 {
		n = rlimit / (math.Log(n) + 0.1544313298)
	}
	maxProduct := uint64(n)

	cache := NewDictMatrixStorage(initialValue, rows, cols)
	m := &CachedMatrixStorage{initialValue, rows, cols, cache, kv, uint64(rlimit), maxProduct}
	logit(fmt.Sprintf("cached matrix store cacheSize = %d, maxProduct = %d, cellSizeInBytes = %d", m.cacheSize, m.maxProduct, cellSizeInBytes), true, DEBUG)
	return m
}

type Uint64Pair struct {
	I uint64
	J uint64
}

func CacheBytesPair(v []byte) Uint64Pair {
	var x Uint64Pair
	x.I = binary.LittleEndian.Uint64(v)
	x.J = binary.LittleEndian.Uint64(v[8:])
	return x
}

func PairToCacheKey(i, j uint64) []byte {
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint64(buf, i)
	binary.LittleEndian.PutUint64(buf[8:], j)
	return buf
}

func CacheBytesToFloat64(v []byte) float64 {
	bits := binary.LittleEndian.Uint64(v)
	float := math.Float64frombits(bits)
	return float
}

func Float64ToCacheBytes(x float64) []byte {
	bits := math.Float64bits(x)
	bytes := make([]byte, 8)
	binary.LittleEndian.PutUint64(bytes, bits)
	return bytes
}

func (m *CachedMatrixStorage) InCache(row, col uint64) bool {
	return row*col < m.maxProduct
}

func (m *CachedMatrixStorage) Get(row, col uint64) float64 {
	if m.InCache(row, col) {
		return m.cache.Get(row, col)
	}
	v, err := m.kv.Get(PairToCacheKey(row, col))
	if err != nil {
		return m.initialValue
	}
	return CacheBytesToFloat64(v)
}

func (m *CachedMatrixStorage) Set(row, col uint64, v float64) {
	if m.InCache(row, col) {
		m.cache.Set(row, col, v)
	} else {
		err := m.kv.Put(PairToCacheKey(row, col), Float64ToCacheBytes(v))
		if err != nil {
			panic(err)
		}
	}
}

func (m *CachedMatrixStorage) Transform(f TransformFunc) {
	m.cache.Transform(f)
	m.kv.Iterate(func(key, val []byte) {
		k := CacheBytesPair(key)
		v := CacheBytesToFloat64(val)
		err := m.kv.Put(key, Float64ToCacheBytes(f(k.I, k.J, v)))
		if err != nil {
			panic(err)
		}
	})
}

func (m *CachedMatrixStorage) ReadOnly() error {
	return m.kv.ReopenReadOnly()
}
