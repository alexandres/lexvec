package lexvecutil

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"strings"
)

type idxUint = uint32

const float64Bytes = 8
const binaryModelMagicNumber = 0xbea25956
const binaryModelVersion = 1
const uint32Bytes = 4

var byteOrder binary.ByteOrder = binary.LittleEndian

type LexVec struct {
	subwordMatrixRows uint32
	dim               uint32
	subwordMinN       uint32
	subwordMaxN       uint32
	vocabSize         uint32
	matrixBaseOffset  int64
	ivWords           []string
	ivWordToIdx       map[string]int
	ivWordToVec       map[string][]float64

	//subword bucket to vec
	subwordsToVec     map[idxUint][]float64
}

func (m LexVec) Dim() uint32 {
	return m.dim
}

func LoadModel(subvecsOutputPath string) (*LexVec, error) {

	out := &LexVec{
		ivWordToIdx:   map[string]int{},
		ivWordToVec:   map[string][]float64{},
		subwordsToVec: map[idxUint][]float64{}, //bucket_id
	}

	f, err := os.Open(subvecsOutputPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	b := make([]byte, float64Bytes)

	magicNumber, err := binaryModelReadUint32(f, b)
	if err != nil {
		return nil, err
	}

	if magicNumber != binaryModelMagicNumber {
		return nil, fmt.Errorf("magic number doesn't match")
	}
	version, err := binaryModelReadUint32(f, b)
	if err != nil {
		return nil, err
	}

	if version != binaryModelVersion {
		return nil, fmt.Errorf("version number doesn't match")
	}

	if vocabSize, err := binaryModelReadUint32(f, b); err != nil {
		return nil, err
	} else {
		out.vocabSize = vocabSize
	}

	if subwordMatrixRows, err := binaryModelReadUint32(f, b); err != nil {
		return nil, err
	} else {
		out.subwordMatrixRows = subwordMatrixRows
	}

	if dim, err := binaryModelReadUint32(f, b); err != nil {
		return nil, err
	} else {
		out.dim = dim
	}

	if subwordMinN, err := binaryModelReadUint32(f, b); err != nil {
		return nil, err
	} else {
		out.subwordMinN = subwordMinN
	}

	if subwordMaxN, err := binaryModelReadUint32(f, b); err != nil {
		return nil, err
	} else {
		out.subwordMaxN = subwordMaxN
	}

	for i := 0; i < int(out.vocabSize); i++ {

		wLen, err := binaryModelReadUint32(f, b)
		if err != nil {
			return nil, err
		}

		b := make([]byte, wLen)
		if _, err := f.Read(b); err != nil {
			return nil, err
		}

		w := string(b)
		out.ivWordToIdx[w] = len(out.ivWords)
		out.ivWords = append(out.ivWords, w)
	}

	if matrixBaseOffset, err := f.Seek(0, 1); err != nil {
		return nil, err
	} else {
		out.matrixBaseOffset = matrixBaseOffset
	}

	for word, idx := range out.ivWordToIdx {
		v := make([]float64, out.dim)
		sumVecFromBin(f, out.matrixBaseOffset, v, idxUint(idx))
		out.ivWordToVec[word] = v
	}

	for i := idxUint(0); i < out.subwordMatrixRows-out.vocabSize; i++ {
		v := make([]float64, out.dim)
		sumVecFromBin(f, out.matrixBaseOffset, v, idxUint(out.vocabSize+i))
		out.subwordsToVec[i] = v
	}

	return out, nil

}

type Result struct {
	Input string
	Word  string
	Vec   []float64
}

func (model LexVec)CalculateOovVectors(input []string) ([]Result, error) {

	out := []Result{}
	for _, line := range input {

		if len(line) == 0 {
			continue
		}

		parts := strings.Split(line, " ")
		w := parts[0]

		res := Result{
			Input: line,
			Word:  w,
			Vec:   make([]float64, model.dim),
		}

		var subwords []string
		if model.subwordMinN > 0 && len(parts) == 1 {
			subwords = computeSubwords(w, int(model.subwordMinN), int(model.subwordMaxN))
		} else {
			subwords = parts[1:]
		}

		var vLen int
		if _, ok := model.ivWordToIdx[w]; ok {
			copy(res.Vec, model.ivWordToVec[w])
			vLen++
		}
		for _, sw := range subwords {
			swBucket, err := subwordBucket(sw, model.subwordMatrixRows-model.vocabSize)
			if err != nil {
				return out, err
			}
			SumVecFromVec(res.Vec, model.subwordsToVec[swBucket])
			vLen++
		}

		//average word and subwords
		if vLen > 0 {
			for j := 0; j < int(model.dim); j++ {
				res.Vec[j] /= float64(vLen)
			}
		}

		out = append(out, res)
	}

	return out, nil
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

func sumVecFromBin(f *os.File, base int64, v []float64, idx idxUint) error {
	dim := len(v)
	if _, err := f.Seek(base+int64(dim)*int64(idx)*float64Bytes, 0); err != nil {
		return err
	}

	b := make([]byte, dim*float64Bytes)
	if _, err := f.Read(b); err != nil {
		return err
	}

	for i := range v {
		z := math.Float64frombits(byteOrder.Uint64(b[float64Bytes*i : float64Bytes*(i+1)]))
		v[i] += z
	}
	return nil
}

func SumVecFromVec(v []float64, from []float64) {
	for i, val := range from {
		v[i] += val
	}
}

func binaryModelReadUint32(f *os.File, b []byte) (uint32, error) {
	_, err := f.Read(b[:uint32Bytes])
	return byteOrder.Uint32(b[:uint32Bytes]), err
}



func subwordBucket(sw string, buckets idxUint) (idxUint,error) {
	h := fnv.New32()
	_, err := h.Write([]byte(sw))
	if err != nil {
		return 0, err
	}

	return h.Sum32() % buckets, nil
}

func subwordIdx(sw string, vocabSize, buckets idxUint) (idxUint, error) {
	subwordBucket, err := subwordBucket(sw, buckets)
	return vocabSize + subwordBucket, err
}
