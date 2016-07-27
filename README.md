# LexVec

This is an implementation of the **LexVec word embedding model** (similar to word2vec and GloVe) that achieves state of the art results in multiple NLP tasks, as described in [this paper](https://arxiv.org/pdf/1606.00819v2) and [this one](https://arxiv.org/pdf/1606.01283v1).

## Pre-trained Vectors

* English Wikipedia 2015 - 3.8B tokens - 303,517 words - 300 dimensions
  - [Word Vectors (327MB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki.300d.W.pos.vectors.gz) 
  - [Word + Context Vectors (351MB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki.300d.W%2bC.pos.vectors.gz)
* English Wikipedia 2015 + [NewsCrawl](http://www.statmt.org/wmt14/translation-task.html) - 7B tokens - 368,999 words - 300 dimensions
  - [Word Vectors (398M)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W.pos.vectors.gz)
  - [Word + Context Vectors (426MB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W%2bC.pos.vectors.gz)

## Installation

### Binary

The easiest way to get started with LexVec is to download the binary release. We only distribute amd64 binaries for Linux.

**[Download binary](https://github.com/alexandres/lexvec/releases)**

If you are using Windows, OS X, 32-bit Linux, or any other OS, follow the instructions below to build from source.

### Building from source

1. [Install the Go compiler](https://golang.org/doc/install)
2. Make sure your `$GOPATH` is set
3. Execute the following commands in your terminal:

   ```bash
   go get github.com/alexandres/lexvec
   cd $GOPATH/src/github.com/alexandres/lexvec
   go build
   ```

## Usage

### In-memory (default, faster)

To get started, run `$ ./demo.sh` which trains a model using the small [text8](http://mattmahoney.net/dc/text8.zip) corpus (100MB from Wikipedia).

Basic usage of LexVec is:

`$ ./lexvec -corpus somecorpus -output someoutputdirectory/vectors`

Run `$ ./lexvec -h` for a full list of options.

Additionally, we provide a `word2vec` script which implements the exact same interface as the [word2vec](https://code.google.com/archive/p/word2vec/) package should you want to test LexVec using existing scripts. 

### External Memory

By default, LexVec stores the sparse matrix being factorized in-memory. This can be a problem if your training corpus is large and your system memory limited. We suggest you first try using the in-memory implementation. If you run into Out-Of-Memory issues, try this External Memory approximation.
xi

`env OUTPUTDIR=output ./external_memory_lexvec.sh -corpus somecorpus -dim 300 ...exactsameoptionsasinmemory`

Pre-processing can be accelerated by installing [nsort](http://www.ordinal.com/try.cgi/nsort-i386-3.4.54.rpm) and [pypy](http://pypy.org/) and editing `pairs_to_counts.sh`.

## References

Salle, A., Idiart, M., & Villavicencio, A. (2016). [Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations](https://arxiv.org/pdf/1606.00819v2). arXiv preprint arXiv:1606.00819.

Salle, A., Idiart, M., & Villavicencio, A. (2016). [Enhancing the LexVec Distributed Word Representation Model Using Positional Contexts and External Memory](https://arxiv.org/pdf/1606.01283v1). arXiv preprint arXiv:1606.01283.

## License

Copyright (c) 2016 Salle, Alexandre <atsalle@inf.ufrgs.br>. All work in this package is distributed under the MIT License.
