# LexVec

This is an implementation of the **LexVec word embedding model** (similar to word2vec and GloVe) that achieves state of the art results in multiple NLP tasks, as described in [this paper](http://anthology.aclweb.org/P16-2068) and [this one](https://arxiv.org/pdf/1606.01283v1).

## Pre-trained Vectors

* [Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz) - 58B tokens - 2,000,000 words - 300 dimensions
  - [Word Vectors (2.2GB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.commoncrawl.300d.W.pos.vectors.gz)
  - [Word + Context Vectors (2.3GB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.commoncrawl.300d.W+C.pos.vectors.gz)

* English Wikipedia 2015 + [NewsCrawl](http://www.statmt.org/wmt14/translation-task.html) - 7B tokens - 368,999 words - 300 dimensions
  - [Word Vectors (398MB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W.pos.vectors.gz)
  - [Word + Context Vectors (426MB)](http://nlpserver2.inf.ufrgs.br/alexandres/vectors/lexvec.enwiki%2bnewscrawl.300d.W%2bC.pos.vectors.gz)

## Evaluation

### In-memory, large corpus

| Model  | GSem | GSyn | MSR | RW | SimLex | SCWS | WS-Sim | WS-Rel | MEN | MTurk | 
| -----  | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| LexVec, Word | **81.1%** | **68.7%** | **63.7%** | **.489** | **.384** | **.652** | .727 | .619 | .759 | .655 | 
| LexVec, Word + Context | 79.3% | 62.6% | 56.4% | .476 | .362 | .629 | .734 | **.663** | **.772** | .649 |
| word2vec Skip-gram | 78.5% | 66.1% | 56.0% | .471 | .347 | .649 | **.774** | .647 | .759 | **.687** |

* All three models were trained using the same English Wikipedia 2015 + NewsCrawl corpus.

* GSem, GSyn, and MSR analogies were solved using [3CosMul](http://www.aclweb.org/anthology/W14-1618).

* LexVec was trained using the default parameters, expanded here for comparison:

  ```
  $ ./lexvec -corpus enwiki+newscrawl.txt -output lexvecvectors -dim 300 -window 2 \
  -subsample 1e-5 -negative 5 -iterations 5 -minfreq 100 -matrix ppmi -model 0
  ```
  
* word2vec Skip-gram was trained using:
  
  ```
  $ ./word2vec -train enwiki+newscrawl.txt -output sgnsvectors -size 300 -window 10 \
  -sample 1e-5 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 5 -min-count 100
  ```

### External memory, huge corpus

| Model  | GSem | GSyn | MSR | RW | SimLex | SCWS | WS-Sim | WS-Rel | MEN | MTurk | 
| -----  | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| LexVec, Word | 76.4% | 71.3% | 70.6% | .508 | **.444** | **.667** | .762 | .668 | .802 | .716 | 
| LexVec, Word + Context | 80.4% | 66.6% | 65.1% | .496 | .419 | .644 | **.775** | **.702** | **.813** | **.712** |
| word2vec | 73.3% | **75.1%** | **75.1%** | **.515** | .436 | .655 | .741 | .610 | .699 | .680 |
| GloVe | **81.8%** | 72.4% | 74.3% | .384 | .374 | .540 | .698 | .571 | .743 | .645 |

* All models use vectors with 300 dimensions.

* GSem, GSyn, and MSR analogies were solved using [3CosMul](http://www.aclweb.org/anthology/W14-1618).

* LexVec was trained using [this release of Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz) 
which contains **58B tokens**, restricting the vocabulary to the 2 million most frequent words, using the following command:

  ```
  $ OUTPUTDIR=output ./external_memory_lexvec.sh -corpus common_crawl.txt -negative 3 \
  -model 0 -maxvocab 2000000 -minfreq 0 -window 2                                             
  ```  
  
* [The pre-trained word2vec vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) were trained using the unreleased Google News corpus containing **100B  tokens**, restricting the vocabulary to the 3 million most frequent words.

* [The pre-trained GloVe vectors](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) were trained using Common Crawl (release unknown) containing **42B  tokens**, restricting the vocabulary to the 1.9 million most frequent words.


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
   $ go get github.com/alexandres/lexvec
   $ cd $GOPATH/src/github.com/alexandres/lexvec
   $ go build
   ```

## Usage

### In-memory (default, faster, more accurate)

To get started, run `$ ./demo.sh` which trains a model using the small [text8](http://mattmahoney.net/dc/text8.zip) corpus (100MB from Wikipedia).

Basic usage of LexVec is:

`$ ./lexvec -corpus somecorpus -output someoutputdirectory/vectors`

Run `$ ./lexvec -h` for a full list of options.

Additionally, we provide a `word2vec` script which implements the exact same interface as the [word2vec](https://code.google.com/archive/p/word2vec/) package should you want to test LexVec using existing scripts. 

### External Memory

By default, LexVec stores the sparse matrix being factorized in-memory. This can be a problem if your training corpus is large and your system memory limited. We suggest you first try using the in-memory implementation, which achieves higher scores in evaluations. If you run into Out-Of-Memory issues, try this External Memory approximation (not as accurate as in-memory; read [paper](https://arxiv.org/pdf/1606.01283v1) for details).

`$ env OUTPUTDIR=output ./external_memory_lexvec.sh -corpus somecorpus -dim 300 ...exactsameoptionsasinmemory`

Pre-processing can be accelerated by installing [nsort](http://www.ordinal.com/try.cgi/nsort-i386-3.4.54.rpm) and [pypy](http://pypy.org/) and editing `pairs_to_counts.sh`.

## References

Salle, Alexandre, Marco Idiart, and Aline Villavicencio. [Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations.](http://anthology.aclweb.org/P16-2068) The 54th Annual Meeting of the Association for Computational Linguistics. 2016.

Salle, A., Idiart, M., & Villavicencio, A. (2016). [Enhancing the LexVec Distributed Word Representation Model Using Positional Contexts and External Memory](https://arxiv.org/pdf/1606.01283v1). arXiv preprint arXiv:1606.01283.

## License

Copyright (c) 2016 Salle, Alexandre <atsalle@inf.ufrgs.br>. All work in this package is distributed under the MIT License.
