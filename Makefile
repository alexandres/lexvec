CC = gcc
CFLAGS = -Ofast -std=gnu99
OBJ = lexvec
BUILD = go build --ldflags '-extldflags "-static"' -o $(OBJ)

optimal:
	CC="$(CC)" CGO_CFLAGS="$(CFLAGS) -march=native" $(BUILD)

cross:
	CC="$(CC)" CGO_CFLAGS="$(CFLAGS)" $(BUILD)
