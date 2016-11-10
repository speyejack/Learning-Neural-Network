SHELL = /bin/sh

CC = g++
CFLAGS = -Wall -Wextra -pedantic -Wwrite-strings
LDFLAGS = -lm
BINARY = learnNN

BUILDDIR = build
SOURCEDIR = src
HEADERDIR = src

SOURCES = $(wildcard $(SOURCEDIR)/*.c)
OBJECTS = $(patsubst $(SOURCEDIR)/%.c, $(BUILDDIR)/%.o, $(SOURCES))

.PHONY: all clean build clear $(BINARY)

all: build

$(BINARY): 
	$(CC) $(CFLAGS) $(LDFLAGS) $(SOURCES) -o $(BINARY) 

clean:
	rm -f $(BINARY) 
	rm -f $(BINARY).exe*
	rm -f $(SOURCEDIR)/*~
	rm -f *.o
	rm -f *~
	rm -f *.stackdump
	rm -f *#

clear :
	clear

rebuild: clean $(BINARY)

retest: rebuild
	./$(BINARY)

debug: clean
	$(CC) $(SOURCES) -o $(BINARY) $(CFLAGS) $(LDFLAGS) -g
	gdb ./$(BINARY)
