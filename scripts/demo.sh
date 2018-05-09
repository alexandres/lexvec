#!/bin/bash

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# build lexvec binary
if [ ! -e "$DIR/../lexvec" ]; then
	pushd
	cd $DIR/.. 
	make
	popd
fi

if [ ! -e text8 ]; then
	echo Downloading text8 corpus
	if hash wget 2>/dev/null; then
		wget http://mattmahoney.net/dc/text8.zip
	else
		curl -O http://mattmahoney.net/dc/text8.zip
	fi
	unzip text8.zip
	rm text8.zip
fi

export OUTPUT=demo_output

mkdir -p $OUTPUT
# These settings are for small corpora such as text8. For larger corpora, stick to the default settings.
$DIR/im_lexvec.sh -corpus text8 -dim 200 -iterations 15 -subsample 1e-4 -window 2 -model 1 -negative 25 -minfreq 5

echo Trained vectors saved to $OUTPUT/vectors.txt and binary model saved to $OUTPUT/model.bin.
