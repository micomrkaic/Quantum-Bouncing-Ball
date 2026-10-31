# Quantum Bouncing Ball
#
#   make              build ./quantum_ball
#   make test         run the built-in numerical self-test (no window needed)
#   make legacy       also build the earlier finite-difference version
#   make install      install to $(PREFIX) (default ~/.local)
#   make uninstall
#   make gif          re-render media/quantum_ball.gif headlessly (needs ffmpeg)
#
# Installed files:
#   $(PREFIX)/bin/quantum_ball
#   $(PREFIX)/share/man/man1/quantum_ball.1
#   $(PREFIX)/share/doc/quantum_ball/{README.md,LICENSE}
#
# System-wide: sudo make install PREFIX=/usr/local

CC      ?= cc
CFLAGS  ?= -O2 -g
WARN     = -Wall -Wextra -pedantic -Wshadow
CSTD     = -std=c11
SDL_CFLAGS := $(shell sdl2-config --cflags 2>/dev/null)
SDL_LIBS   := $(shell sdl2-config --libs 2>/dev/null || echo -lSDL2)
LDLIBS   = $(SDL_LIBS) -lfftw3 -lm

PREFIX  ?= $(HOME)/.local
BINDIR  ?= $(PREFIX)/bin
MANDIR  ?= $(PREFIX)/share/man/man1
DOCDIR  ?= $(PREFIX)/share/doc/quantum_ball
INSTALL ?= install

TARGET = quantum_ball

.PHONY: all test legacy install uninstall gif clean

all: $(TARGET)

$(TARGET): quantum_ball.c
	$(CC) $(CSTD) $(WARN) $(CFLAGS) $(SDL_CFLAGS) $< -o $@ $(LDLIBS)

# The 1.x finite-difference / RK4 version, kept for comparison (SDL2 only).
legacy: legacy/quantum_ball_rk4
legacy/quantum_ball_rk4: legacy/quantum_ball_rk4.c
	$(CC) $(CSTD) -O2 $(SDL_CFLAGS) $< -o $@ $(SDL_LIBS) -lm

test: $(TARGET)
	./$(TARGET) --selftest

gif: $(TARGET)
	rm -rf build/frames && mkdir -p build/frames media
	SDL_VIDEODRIVER=dummy ./$(TARGET) --mode quantum --frames 300 --record build/frames
	ffmpeg -v error -y -framerate 60 -i build/frames/frame_%05d.ppm \
	  -vf "fps=15,crop=1080:720:320:90,scale=600:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none" \
	  -frames:v 60 media/quantum_ball.gif
	rm -rf build/frames

install: $(TARGET)
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(MANDIR) $(DESTDIR)$(DOCDIR)
	$(INSTALL) -m 755 $(TARGET) $(DESTDIR)$(BINDIR)/$(TARGET)
	$(INSTALL) -m 644 quantum_ball.1 $(DESTDIR)$(MANDIR)/quantum_ball.1
	$(INSTALL) -m 644 README.md LICENSE $(DESTDIR)$(DOCDIR)/
	@case ":$$PATH:" in *":$(BINDIR):"*) ;; *) echo "note: $(BINDIR) is not in your PATH" ;; esac

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/$(TARGET)
	rm -f $(DESTDIR)$(MANDIR)/quantum_ball.1
	rm -rf $(DESTDIR)$(DOCDIR)

clean:
	rm -rf $(TARGET) legacy/quantum_ball_rk4 build
