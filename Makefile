CC = gcc
CFLAGS = -Ofast -std=gnu99
OBJ = lexvec
BUILD = go build --ldflags '-extldflags "-static"' -o $(OBJ)

optimal:
	cd cmd && CC="$(CC)" CGO_CFLAGS="$(CFLAGS) -march=native" $(BUILD) && cd -

cross:
	cd cmd && CC="$(CC)" CGO_CFLAGS="$(CFLAGS)" $(BUILD) && cd -
