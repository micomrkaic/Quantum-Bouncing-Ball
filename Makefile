CC ?= gcc
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -pedantic
SDL_CFLAGS := $(shell sdl2-config --cflags)
SDL_LIBS   := $(shell sdl2-config --libs)
FFTW_LIBS  := -lfftw3f
LDLIBS     := $(SDL_LIBS) $(FFTW_LIBS) -lm

TARGET = quantum_ball_pseudospectral
SRC    = quantum_ball_pseudospectral.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SDL_CFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
