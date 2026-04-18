# Quantum Ball in a 2D Box

An SDL2 + FFTW demo that compares two versions of the same ballistic problem inside a rectangular box:

- **Classical mode**: a point mass under constant gravity with elastic wall bounces
- **Quantum mode**: a 2D Gaussian wave packet evolved with a **split-step pseudospectral solver** for the time-dependent Schrödinger equation with hard-wall boundary conditions

The goal is not physical realism in SI units. The program uses normalized units chosen for a visually useful interactive demo.

## Features

- Interactive control panel with sliders for:
  - launch angle
  - initial speed
  - gravity
  - box width
  - box height
- Toggle between **Classical** and **Quantum** modes
- Pause and reset controls
- Hard-wall rectangular box
- Quantum evolution with a **sine pseudospectral basis** implemented using **FFTW DST-I** transforms

## Mathematical model

### Classical mode

The classical ball satisfies

\[
\dot x = v_x, \qquad \dot y = v_y,
\]
\[
\dot v_x = 0, \qquad \dot v_y = g,
\]

with perfectly elastic reflections at the walls.

The coordinate convention is screen-like:

- \(x\) increases to the right
- \(y\) increases downward
- positive gravity accelerates the ball downward

### Quantum mode

The quantum mode solves

\[
i\hbar \frac{\partial \psi}{\partial t}
=
\left[-\frac{\hbar^2}{2m}\Delta + V(x,y)\right]\psi
\]

inside a rectangle with hard walls,

\[
\psi = 0
\]

on the boundary.

Because \(y\) increases downward in this program, the gravitational potential consistent with downward acceleration is

\[
V(x,y) = -mgy.
\]

That sign matters. If the sign is flipped, the quantum packet accelerates upward and appears to “float.”

## Numerical method

The quantum solver uses **Strang split-step evolution**:

\[
\psi^{n+1}
\approx
\exp\!\left(-\frac{i}{\hbar}V\frac{\Delta t}{2}\right)
\exp\!\left(-\frac{i}{\hbar}T\Delta t\right)
\exp\!\left(-\frac{i}{\hbar}V\frac{\Delta t}{2}\right)
\psi^n.
\]

### Why sine transforms?

The box has **Dirichlet boundary conditions** (hard walls), so a sine basis is the natural spectral basis:

\[
\sin\left(\frac{n\pi x}{L_x}\right)
\sin\left(\frac{m\pi y}{L_y}\right).
\]

In that basis:

- the kinetic operator is diagonal
- the hard-wall boundary conditions are built in automatically
- the forward and inverse transforms can be implemented with FFTW's **DST-I**

### Kinetic step

In spectral space, each sine mode picks up a phase factor

\[
\exp\left(-\frac{i}{\hbar} T_{nm} \Delta t\right),
\]

where

\[
T_{nm} = \frac{\hbar^2}{2m}
\left[
\left(\frac{n\pi}{L_x}\right)^2 +
\left(\frac{m\pi}{L_y}\right)^2
\right].
\]

### Potential step

The potential step is applied pointwise in physical space:

\[
\psi \leftarrow
\exp\left(-\frac{i}{\hbar}V\frac{\Delta t}{2}\right) \psi.
\]

## Files

- `quantum_ball_pseudospectral.c` — main program
- `Makefile` — build rules

## Dependencies

You need:

- **SDL2**
- **FFTW3 single precision** (`fftw3f`)
- a C compiler such as `gcc` or `clang`

### Debian / Ubuntu

```bash
sudo apt install libsdl2-dev libfftw3-dev build-essential
```

### Fedora

```bash
sudo dnf install SDL2-devel fftw-devel gcc make
```

### Arch

```bash
sudo pacman -S sdl2 fftw base-devel
```

### macOS (Homebrew)

```bash
brew install sdl2 fftw
```

## Build

Use the provided Makefile:

```bash
make
```

That should produce:

```bash
./quantum_ball_pseudospectral
```

To clean build artifacts:

```bash
make clean
```

## Run

```bash
./quantum_ball_pseudospectral
```

## Controls

### Mouse

- Drag sliders to change parameters
- Click **CLASSICAL** or **QUANTUM** to switch modes
- Click **RESET** to restart from the current parameters
- Click **PAUSE** to pause/resume

### Keyboard

- `C` — classical mode
- `Q` — quantum mode
- `R` — reset
- `Space` — pause/resume
- `Esc` — quit

## Interpreting the display

### Classical mode

A single ball follows the usual ballistic path and reflects elastically off the walls.

### Quantum mode

The colored field shows the wave-packet amplitude distribution. The white dot marks the expectation-value position

\[
(\langle x \rangle, \langle y \rangle).
\]

This is not a rigid ball. The packet:

- spreads
- interferes with itself after reflections
- can deviate from the naive classical point-particle path

That behavior is expected.

## Why the quantum and classical trajectories do not match exactly

Even with the gravity sign fixed, the quantum packet is not a classical projectile.

Reasons:

- the packet has finite spatial width
- it has momentum uncertainty
- it disperses over time
- wall reflections distort the packet
- the center of mass only approximately tracks the classical path in a semiclassical regime

If you want the quantum packet to look more classical, push the parameters toward a more semiclassical regime, for example by:

- increasing `mass`
- decreasing `hbar`
- using a sufficiently narrow momentum spread relative to the packet momentum

In the current code, the default values are deliberately simple rather than tuned for maximum classical agreement.

## Implementation notes

- The quantum solver uses FFTW **real-to-real** transforms with `FFTW_RODFT00`, which corresponds to **DST-I**.
- The transform is applied first across rows and then across columns.
- The code normalizes the wavefunction after each full quantum update.
- The current implementation allocates temporary buffers during the kinetic step and inverse transform. That is acceptable for a demo but can be optimized.

## Known caveats

- Units are normalized and chosen for visualization, not for laboratory realism.
- The classical and quantum modes are conceptually comparable, but not expected to coincide exactly.
- The quantum packet can look “too quantum” unless `mass`, `hbar`, packet width, and time step are tuned more carefully.
- The code is an interactive demonstration, not a production PDE package.

## Possible extensions

- add trajectory traces
- expose `mass`, `hbar`, and packet width as sliders
- add a “semiclassical” preset
- replace temporary heap allocations with persistent work arrays
- add phase-color rendering in quantum mode
- add absorbing layers or softer walls

## License

No license has been attached yet. Add one explicitly if you want to distribute the code.
