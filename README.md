# LexVec

This is an implementation of the **LexVec word embedding model** (similar to word2vec and GloVe) that achieves state of the art results in multiple NLP tasks, as described in [these papers](#refs).

## Pre-trained Vectors

### Subword LexVec
* [Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz) - 58B tokens, cased - 2,000,000 words - 300 dimensions
  - [Word Vectors (2.1GB)](https://www.dropbox.com/s/mrxn933chn5u37z/lexvec.ngramsubwords.300d.W.pos.vectors.gz?dl=1)
  - [Binary model (8.6GB)](https://www.dropbox.com/s/buix0deqlks4312/lexvec.ngramsubwords.300d.W.pos.bin.gz?dl=1) - use this to [compute vectors for out-of-vocabulary (OOV) words](#oov)

### LexVec

* [Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz) - 58B tokens, lowercased - 2,000,000 words - 300 dimensions
  - [Word Vectors (2.2GB)](https://www.dropbox.com/s/flh1fjynqvdsj4p/lexvec.commoncrawl.300d.W.pos.vectors.gz?dl=1)
  - [Word + Context Vectors (2.3GB)](https://www.dropbox.com/s/zkiajh6fj0hm0m7/lexvec.commoncrawl.300d.W%2BC.pos.vectors.gz?dl=1)

* English Wikipedia 2015 + [NewsCrawl](http://www.statmt.org/wmt14/translation-task.html) - 7B tokens - 368,999 words - 300 dimensions
  - [Word Vectors (398MB)](https://www.dropbox.com/s/kguufyc2xcdi8yk/lexvec.enwiki%2Bnewscrawl.300d.W.pos.vectors.gz?dl=1)
  - [Word + Context Vectors (426MB)](https://www.dropbox.com/s/u320t9bw6tzlwma/lexvec.enwiki%2Bnewscrawl.300d.W%2BC.pos.vectors.gz?dl=1)

## Evaluation: Subword LexVec

### External memory, huge corpus

| Model  | GSem | GSyn | MSR | RW | SimLex | SCWS | WS-Sim | WS-Rel | MEN | MTurk | 
| -----  | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| LexVec | 72.6% | **73.8%** | **73.2%** | **.539** | **.477** | **.687** | .809 | .696 | **.814** | **.717** |
| fastText | **75.0%** | 72.1% | 71.8% | .522 | .424 | .673 | **.810** | **.724** | .805 | **.717** |

* Both models use vectors with 300 dimensions.

* Both models use character n-grams of length 3-6 as subwords.

* All tasks are evaluated using cased words (``"Toronto" != "toronto"``).

* GSem, GSyn, and MSR analogies were solved using [3CosMul](http://www.aclweb.org/anthology/W14-1618).

* Both models were trained using [this release of Common Crawl](http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/en-new.xz) 
which contains **58B tokens**, restricting the vocabulary to the 2 million most frequent cased words.

* Subword LexVec was trained using the following command:

  ```
  $ OUTPUT=output scripts/em_lexvec.sh -corpus common_crawl_uncased.txt -negative 3 -dim 300 -subsample 1e-5 -minfreq 0 -window 2 -minn 3 -maxn 6
  ```  
  
* fastText was trained using the following command:

  ```
  $ ./fasttext skipgram -input common_crawl_cased.txt -minCount 0 -t 1e-5 -dim 300 -lr 0.025 -minn 3 -maxn 6                  
  ```

## Evaluation: LexVec

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
  $ OUTPUT=output scripts/im_lexvec.sh -corpus enwiki+newscrawl.txt -dim 300 -window 2 -subsample 1e-5 -negative 5 -iterations 5 -minfreq 100 -model 0 -minn 0
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
  $ OUTPUT=output scripts/em_lexvec.sh -corpus common_crawl.txt -negative 3 -dim 300 -subsample 1e-5 -minfreq 0 -window 2 -minn 0                                     
  ```  
  
* [The pre-trained word2vec vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing) were trained using the unreleased Google News corpus containing **100B  tokens**, restricting the vocabulary to the 3 million most frequent words.

* [The pre-trained GloVe vectors](http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip) were trained using Common Crawl (release unknown) containing **42B  tokens**, restricting the vocabulary to the 1.9 million most frequent words.


## Installation

1. [Install the Go compiler](https://golang.org/doc/install) and clang.
2. Make sure your `$GOPATH` is set
3. Execute the following commands in your terminal:

   ```bash
   $ go get github.com/alexandres/lexvec
   $ cd $GOPATH/src/github.com/alexandres/lexvec
   $ make
   ```

## Usage

### Training

#### In-memory

To get started, run `$ scripts/demo.sh` which trains a model using the small [text8](http://mattmahoney.net/dc/text8.zip) corpus (100MB from Wikipedia).

Basic usage of LexVec is:

`$ OUTPUT=dirwheretostorevectors scripts/im_lexvec.sh -corpus somecorpus`

Run `$ ./lexvec -h` for a full list of options.

#### External Memory

By default, LexVec stores the sparse matrix being factorized in-memory. This can be a problem if your training corpus is large and your system memory limited. We suggest you first try using the in-memory implementation. If you run into Out-Of-Memory issues, use the External Memory variant with the ``-memory`` option specifying how many GBs of memory to use for the sort buffer.

`$ OUTPUT=dirwheretostorevectors scripts/em_lexvec.sh -corpus somecorpus -memory 4. ...exactsameoptionsasinmemory`

### Subword LexVec

#### Training

Subword information is controlled by the options ``-minn``, ``-maxn``, and ``-subword``.

* To disable the use of subword information, specify ``-minn 0``.

* To use character n-grams of length 3-6, specify ``-minn 3 -maxn 6`` (this is the default configuration).

* To provide your own subword information (such as morphological segmentation), specify ``-minn 0 -subword subwords.txt``, where the subwords file contains one line for each vocabulary word (vocabulary must match that of ``-vocab``), each line containing a word followed by each of its subwords, separated by spaces.

By default, the binary model used for computing OOV word vectors is saved to ``$OUTPUT/model.bin``. Set ``-outputsub ""`` to disable saving this model.

#### <a name="oov"></a> Computing vectors for OOV words

Use the binary model to compute vector for OOV words:

* Using the Go executable by providing one word per line on ``stdin``:

  ``$ echo "marvelicious" | ./lexvec embed -outputsub pathtomodel.bin``

* Using the [Python lib](https://github.com/alexandres/lexvec/blob/master/python):

  ```python
  import lexvec
  model = lexvec.Model('pathtomodel.bin')
  vector = model.word_rep('marvelicious')
  ```

*Note: You can also use these commands to get vectors for in-vocabulary words as the binary model stores the vocabulary used for training.*

## <a name="refs"></a> References

Alexandre Salle and Aline Villavicencio. "Incorporating Subword Information into Matrix Factorization Word Embeddings." *to appear at* Second Workshop on Subword and Character LEvel Models in NLP (2018). [(preprint)](https://arxiv.org/pdf/1805.03710)

Alexandre Salle, Marco Idiart, and Aline Villavicencio. "Enhancing the LexVec Distributed Word Representation Model Using Positional Contexts and External Memory." arXiv preprint arXiv:1606.01283 (2016). [(pdf)](https://arxiv.org/pdf/1606.01283v1)

Alexandre Salle, Marco Idiart, and Aline Villavicencio. "Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations." ACL (2016). [(pdf)](http://anthology.aclweb.org/P16-2068)

## License

Copyright (c) 2016-2018 Salle, Alexandre <alex@alexsalle.com>. All work in this package is distributed under the MIT License.
