CCFLAGS = -Wall -Wshadow -O2 -g -fpermissive
LFLAGS = -lm

all: ADALINE

test: ADALINE.o
	$(CC) $(CCFLAGS) -o $@ $^ $(LFLAGS)
	./$@

.c.o:
	$(CC) -c $(CCFLAGS) $< -o $@


clean:
	rm *.o
	rm *.exe
	rm persist.txt
