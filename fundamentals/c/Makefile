# Compiler
CC = gcc
# Compiler flags
CFLAGS = -g -O0 -W -Ipalp -lm

# Source files
SRCS := src/GA.c src/Misc.c src/fitness.c src/Generation.c src/population.c palp/Coord.c palp/LG.c palp/Vertex.c palp/Rat.c palp/Polynf.c

# Target executable
TARGET = GA

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f $(TARGET)

.PHONY: all clean
