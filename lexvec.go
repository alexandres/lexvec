package main

import (
	"math/rand"
	"os"
	"path"
	"strings"
)

/*
lexvec vocab -vocab output/vocab.txt -output output/vectors.txt -outputsub output/model.bin -corpus /mnt/c/Users/starsc/Downloads/semantic.txt -dim 300 -window 2 -subsample 1e-5 -negative 5 -iterations 5 -minfreq 100 -model 0 -minn 0
lexvec train -vocab output/vocab.txt -output output/vectors.txt -outputsub output/model.bin -corpus /mnt/c/Users/starsc/Downloads/semantic.txt -dim 300 -window 2 -subsample 1e-5 -negative 5 -iterations 5 -minfreq 100 -model 0 -minn 0
*/

type OovVectors map[string][]float64

func GetOovVectors(words []string, subvecsOutput *os.File) (OovVectors, error) {
	var (
		err error
		ov  = make(OovVectors, float64Bytes)
		b   = make([]byte, float64Bytes)

		magicNumber       = binaryModelReadUint32(subvecsOutput, b)
		version           = binaryModelReadUint32(subvecsOutput, b)
		vocabSize         = binaryModelReadUint32(subvecsOutput, b)
		subwordMatrixRows = binaryModelReadUint32(subvecsOutput, b)
		dim               = binaryModelReadUint32(subvecsOutput, b)
		subwordMinN       = binaryModelReadUint32(subvecsOutput, b)
		subwordMaxN       = binaryModelReadUint32(subvecsOutput, b)

		matrixBaseOffset int64
	)

	if magicNumber != binaryModelMagicNumber {
		logln(errorLogLevel, "magic number doesnt match")
	}
	if version != binaryModelVersion {
		logln(errorLogLevel, "version number doesnt match")
	}

	var ivWords []string
	ivWordToIdx := make(map[string]int)
	for i := 0; i < int(vocabSize); i++ {
		wLen := binaryModelReadUint32(subvecsOutput, b)

		b := make([]byte, wLen)
		if _, err = subvecsOutput.Read(b); err != nil {
			return ov, nil
		}

		w := string(b)
		ivWordToIdx[w] = len(ivWords)
		ivWords = append(ivWords, w)
	}
	if matrixBaseOffset, err = subvecsOutput.Seek(0, 1); err != nil {
		return ov, nil
	}

	for _, oov := range words {
		vec := make([]float64, dim)
		if len(oov) == 0 {
			break
		}
		parts := strings.Split(oov, " ")
		w := parts[0]
		var subwords []string
		if subwordMinN > 0 && len(parts) == 1 {
			subwords = computeSubwords(w, int(subwordMinN), int(subwordMaxN))
		} else {
			subwords = parts[1:]
		}
		for j := 0; j < int(dim); j++ {
			vec[j] = 0
		}
		var vLen int
		if idx, ok := ivWordToIdx[w]; ok {
			sumVecFromBin(subvecsOutput, matrixBaseOffset, vec, idxUint(idx))
			vLen++
		}
		for _, sw := range subwords {
			sumVecFromBin(subvecsOutput, matrixBaseOffset, vec, subwordIdx(sw, vocabSize, subwordMatrixRows-vocabSize))
			vLen++
		}
		if vLen > 0 {
			for j := 0; j < int(dim); j++ {
				vec[j] /= float64(vLen)
			}
		}
		ov[oov] = vec
	}
	return ov, nil
}

func StartTrain(outputFolder, corpusP string,
	dimP idxUint, subsampleP real,
	minfreqP countUint,
	modelP, windowP, negativeP, iterationsP, subwordMinNP int) {
	if _, err := os.Stat(outputFolder); os.IsNotExist(err) {
		_ = os.Mkdir(outputFolder, 0644)
	}

	randng = rand.New(rand.NewSource(1))
	vocabPath = path.Join(outputFolder, "vocab.txt")
	vectorOutputPath = path.Join(outputFolder, "vectors.txt")
	subvecsOutputPath = path.Join(outputFolder, "model.bin")
	corpusPath = corpusP
	dim = dimP
	subsample = subsampleP
	window = windowP
	negative = negativeP
	iterations = iterationsP
	minFreq = minfreqP
	model = modelP
	subwordMinN = subwordMinNP

	buildVocab()
	saveVocab()

	readVocab()
	processSubwords()
	buildCoocMatrix()
	calculateCdsTotalAndLogs()
	initModel()
	train(newTrainIteratorIM())
	saveVectors()
}
