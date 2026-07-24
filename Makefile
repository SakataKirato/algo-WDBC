CC = cc
CFLAGS = -std=c17 -O2 -Wall -Wextra -Wpedantic
LDLIBS = -lm
feature_select: feature_select.c
	$(CC) $(CFLAGS) -o $@ feature_select.c $(LDLIBS)

clean:
	rm -f feature_select main.o data.o evaluate.o search.o

.PHONY: clean
