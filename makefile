SHELL = /bin/sh

CC = g++
CFLAGS = -Wall -Wextra -pedantic -Wwrite-strings -std=c++11
OPTFLAG= -O3
LDFLAGS = -lm
BINARY = learnNN

BUILDDIR = build
SOURCEDIR = src
HEADERDIR = src

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))

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
	$(CC) $(SOURCES) -o $(BINARY) $(CFLAGS) $(LDFLAGS) -ggdb
	gdb ./$(BINARY)

memCheck: clean
	$(CC) $(SOURCES) -o $(BINARY) $(CFLAGS) $(LDFLAGS) -ggdb
	valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./$(BINARY)

optimize: clean 
	$(CC) $(CFLAGS) $(LDFLAGS) $(SOURCES) -o $(BINARY) $(OPTFLAG)

reoptimize: clean optimize
	./$(BINARY)
