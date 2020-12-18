package main

import (
	"os"
	"reflect"
	"testing"
)

func TestGetOovVectors(t *testing.T) {
	f, _ := os.Open("output/model.bin")

	type args struct {
		words         []string
		subvecsOutput *os.File
	}
	tests := []struct {
		name    string
		args    args
		want    OovVectors
		wantErr bool
	}{
		{
			name: "test",
			args: args{
				words:         []string{"test", "model"},
				subvecsOutput: f,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetOovVectors(tt.args.words, tt.args.subvecsOutput)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetOovVectors() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetOovVectors() got = %v, want %v", got, tt.want)
			}
			t.Log(got)
		})
	}
}

func TestStartTrain(t *testing.T) {
	type args struct {
		outputFolder string
		corpus       string
		dim          idxUint
		subsample    real
		window       int
		negative     int
		iterations   int
		minfreq      countUint
		model        int
		subwordMinN  int
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "test",
			args: args{
				outputFolder: "output",
				corpus:       "morphological.txt",
				dim:          300,
				subsample:    1e-5,
				window:       2,
				negative:     5,
				iterations:   5,
				minfreq:      100,
				model:        0,
				subwordMinN:  0,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			StartTrain(tt.args.outputFolder, tt.args.corpus, tt.args.dim, tt.args.subsample, tt.args.minfreq, tt.args.model, tt.args.window, tt.args.negative, tt.args.iterations, tt.args.subwordMinN)
		})
	}
}
