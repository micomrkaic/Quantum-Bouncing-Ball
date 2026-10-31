/*
    quantum_ball.c

    Classical / quantum ballistic ball in a 2D rectangular box with gravity,
    plus the stationary states of the quantum bouncer.

    Classical mode:
        - point mass, constant gravity, elastic bounces at the walls

    Quantum mode:
        - 2D time-dependent Schrodinger equation in a hard-wall rectangle
        - Strang split-step evolution
        - sine pseudospectral basis via FFTW DST-I transforms (one 2D plan, double precision)

    Eigenstate mode:
        - stationary states psi_{n,m}(x,y) = sin(n pi x/Lx) phi_m(y), where
          phi_m solves the 1D problem by Numerov shooting; for strong gravity
          these are the Airy states of the quantum bouncer

    Conventions (screen-like): y increases DOWNWARD, gravity accelerates the
    ball downward, so the potential is

        V(x, y) = - m g y

    and the floor of the bouncer is the bottom wall y = Ly.

    PDE:
        i hbar dpsi/dt = [ -(hbar^2/(2m)) Laplacian - m g y ] psi,   psi = 0 on the walls

    One Strang step:
        psi^{n+1} = exp(-i V dt / 2hbar) exp(-i T dt / hbar) exp(-i V dt / 2hbar) psi^n

    Units are normalized (hbar = m = 1); the sliders choose a visually useful regime.

    Build: see the Makefile, or
        gcc -O2 -std=c11 quantum_ball.c -o quantum_ball $(sdl2-config --cflags --libs) -lfftw3 -lm
*/

#include <SDL2/SDL.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VERSION "2.0"

#define WINDOW_W 1400
#define WINDOW_H 900
#define CONTROL_W 320
#define MARGIN 16

#define CLAMP(x,a,b) ((x)<(a)?(a):((x)>(b)?(b):(x)))
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_EIGEN_M 24          /* highest vertical quantum number offered */
#define MAX_EIGEN_N 12

typedef enum { MODE_CLASSICAL = 0, MODE_QUANTUM = 1, MODE_EIGEN = 2 } SimMode;

typedef struct {
    float angle_deg;
    float speed;
    float gravity;
    float box_w;
    float box_h;
} Params;

typedef struct {
    float x, y;
    float vx, vy;
    float radius;
} ClassicalBall;

typedef struct {
    SDL_Rect rect;
    float minv, maxv;
    float *value;
    const char *label;
    bool dragging;
} Slider;

typedef struct {
    SDL_Rect rect;
    const char *label;
} Button;

/* The wave function is stored on the interior grid only (mx x my points,
   row-major, index j*mx + i); psi is identically zero on the walls, which
   is what DST-I assumes. x_i = (i+1) dx, y_j = (j+1) dy. */
typedef struct {
    int mx, my;
    float Lx, Ly;
    double dx, dy;
    double dt;
    double mass;
    double hbar;

    double *re, *im;            /* psi, interior, size mx*my (double: keeps the
                                   norm and energy drift at roundoff level) */
    double *kin_phase_re;       /* exp(-i T dt / hbar) per sine mode */
    double *kin_phase_im;
    double *kin_energy;         /* T_nm per sine mode */
    double *scratch_re, *scratch_im;  /* for <T> at display time */
    fftw_plan dst2;            /* 2D DST-I, in place; its own inverse up to a scale */

    long   steps;
} QuantumField;

typedef struct {
    int n, m;                   /* quantum numbers, 1-based */
    double energy;              /* E_{n,m} */
    double energy_airy;         /* bouncer approximation for the vertical part + sine part */
    double *phi;                /* phi_m on the interior y grid, size my, normalised */
    int my;
    double Ly;
    int nodes;
} EigenState;

/* ----------------------------- Tiny bitmap font ----------------------------- */

typedef struct {
    char c;
    unsigned char rows[7];
} Glyph;

static const Glyph g_font[] = {
    {'A',{0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'B',{0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}},
    {'C',{0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}},
    {'D',{0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}},
    {'E',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}},
    {'F',{0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}},
    {'G',{0x0E,0x11,0x10,0x17,0x11,0x11,0x0F}},
    {'H',{0x11,0x11,0x11,0x1F,0x11,0x11,0x11}},
    {'I',{0x1F,0x04,0x04,0x04,0x04,0x04,0x1F}},
    {'J',{0x07,0x02,0x02,0x02,0x12,0x12,0x0C}},
    {'K',{0x11,0x12,0x14,0x18,0x14,0x12,0x11}},
    {'L',{0x10,0x10,0x10,0x10,0x10,0x10,0x1F}},
    {'M',{0x11,0x1B,0x15,0x15,0x11,0x11,0x11}},
    {'N',{0x11,0x11,0x19,0x15,0x13,0x11,0x11}},
    {'O',{0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}},
    {'P',{0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}},
    {'Q',{0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}},
    {'R',{0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}},
    {'S',{0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}},
    {'T',{0x1F,0x04,0x04,0x04,0x04,0x04,0x04}},
    {'U',{0x11,0x11,0x11,0x11,0x11,0x11,0x0E}},
    {'V',{0x11,0x11,0x11,0x11,0x11,0x0A,0x04}},
    {'W',{0x11,0x11,0x11,0x15,0x15,0x15,0x0A}},
    {'X',{0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}},
    {'Y',{0x11,0x11,0x0A,0x04,0x04,0x04,0x04}},
    {'Z',{0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}},
    {'0',{0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}},
    {'1',{0x04,0x0C,0x14,0x04,0x04,0x04,0x1F}},
    {'2',{0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}},
    {'3',{0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}},
    {'4',{0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}},
    {'5',{0x1F,0x10,0x10,0x1E,0x01,0x01,0x1E}},
    {'6',{0x0E,0x10,0x10,0x1E,0x11,0x11,0x0E}},
    {'7',{0x1F,0x01,0x02,0x04,0x08,0x08,0x08}},
    {'8',{0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}},
    {'9',{0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E}},
    {':',{0x00,0x04,0x04,0x00,0x04,0x04,0x00}},
    {'.',{0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}},
    {'-',{0x00,0x00,0x00,0x1F,0x00,0x00,0x00}},
    {'+',{0x00,0x04,0x04,0x1F,0x04,0x04,0x00}},
    {'<',{0x02,0x04,0x08,0x10,0x08,0x04,0x02}},
    {'>',{0x08,0x04,0x02,0x01,0x02,0x04,0x08}},
    {'=',{0x00,0x1F,0x00,0x1F,0x00,0x00,0x00}},
    {'/',{0x01,0x02,0x04,0x08,0x10,0x00,0x00}},
    {'(',{0x02,0x04,0x08,0x08,0x08,0x04,0x02}},
    {')',{0x08,0x04,0x02,0x02,0x02,0x04,0x08}},
    {',',{0x00,0x00,0x00,0x00,0x0C,0x04,0x08}},
    {' ',{0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
};

static const Glyph *find_glyph(char c) {
    size_t n = sizeof(g_font) / sizeof(g_font[0]);
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    for (size_t i = 0; i < n; ++i) if (g_font[i].c == c) return &g_font[i];
    return &g_font[n - 1];
}

static void draw_char(SDL_Renderer *r, int x, int y, int s, char c, SDL_Color col) {
    const Glyph *g = find_glyph(c);
    SDL_SetRenderDrawColor(r, col.r, col.g, col.b, col.a);
    for (int row = 0; row < 7; ++row) {
        unsigned char bits = g->rows[row];
        for (int colb = 0; colb < 5; ++colb) {
            if (bits & (1 << (4 - colb))) {
                SDL_Rect px = {x + colb*s, y + row*s, s, s};
                SDL_RenderFillRect(r, &px);
            }
        }
    }
}

static void draw_text(SDL_Renderer *r, int x, int y, int s, const char *text, SDL_Color col) {
    for (const char *p = text; *p; ++p, x += 6*s) draw_char(r, x, y, s, *p, col);
}

/* --------------------------------- UI ------------------------------------- */

static bool point_in_rect(int x, int y, SDL_Rect rc) {
    return x >= rc.x && x < rc.x + rc.w && y >= rc.y && y < rc.y + rc.h;
}

static void draw_slider(SDL_Renderer *r, const Slider *s) {
    SDL_Color fg = {235,235,235,255};
    SDL_Color track = {90,90,100,255};
    SDL_Color knob = {220,180,70,255};
    char buf[128];
    snprintf(buf, sizeof buf, "%s %.2f", s->label, *s->value);
    draw_text(r, s->rect.x, s->rect.y - 22, 2, buf, fg);
    SDL_Rect bar = {s->rect.x, s->rect.y + s->rect.h/2 - 3, s->rect.w, 6};
    SDL_SetRenderDrawColor(r, track.r, track.g, track.b, track.a);
    SDL_RenderFillRect(r, &bar);
    float t = (*s->value - s->minv)/(s->maxv - s->minv);
    t = CLAMP(t, 0.0f, 1.0f);
    int kx = s->rect.x + (int)(t * s->rect.w);
    SDL_Rect k = {kx - 7, s->rect.y, 14, s->rect.h};
    SDL_SetRenderDrawColor(r, knob.r, knob.g, knob.b, knob.a);
    SDL_RenderFillRect(r, &k);
}

static void slider_handle_mouse(Slider *s, int mx) {
    float t = (float)(mx - s->rect.x) / (float)s->rect.w;
    t = CLAMP(t, 0.0f, 1.0f);
    *s->value = s->minv + t * (s->maxv - s->minv);
}

static void draw_button(SDL_Renderer *r, const Button *b, bool active) {
    SDL_Color edge = {180,180,180,255};
    SDL_Color fill = active ? (SDL_Color){70,120,70,255} : (SDL_Color){70,70,85,255};
    SDL_Color fg = {240,240,240,255};
    SDL_SetRenderDrawColor(r, fill.r, fill.g, fill.b, fill.a);
    SDL_RenderFillRect(r, &b->rect);
    SDL_SetRenderDrawColor(r, edge.r, edge.g, edge.b, edge.a);
    SDL_RenderDrawRect(r, &b->rect);
    draw_text(r, b->rect.x + 10, b->rect.y + 8, 2, b->label, fg);
}

/* --------------------------- Classical simulation -------------------------- */

static void classical_reset(ClassicalBall *b, const Params *p) {
    float theta = p->angle_deg * (float)M_PI / 180.0f;
    b->radius = 0.03f * fminf(p->box_w, p->box_h);
    b->x = 0.15f * p->box_w;
    b->y = 0.20f * p->box_h;
    b->vx = p->speed * cosf(theta);
    b->vy = -p->speed * sinf(theta);
}

static void classical_step(ClassicalBall *b, const Params *p, float dt) {
    b->vy += p->gravity * dt;
    b->x += b->vx * dt;
    b->y += b->vy * dt;
    if (b->x - b->radius < 0.0f) { b->x = b->radius; b->vx = -b->vx; }
    if (b->x + b->radius > p->box_w) { b->x = p->box_w - b->radius; b->vx = -b->vx; }
    if (b->y - b->radius < 0.0f) { b->y = b->radius; b->vy = -b->vy; }
    if (b->y + b->radius > p->box_h) { b->y = p->box_h - b->radius; b->vy = -b->vy; }
}

/* Energy per unit mass with V = -g y (the same convention as the quantum side). */
static double classical_energy(const ClassicalBall *b, const Params *p) {
    return 0.5 * ((double)b->vx*b->vx + (double)b->vy*b->vy) - (double)p->gravity * b->y;
}

/* ---------------------------- Quantum simulation --------------------------- */

static void quantum_free(QuantumField *q) {
    if (q->dst2) fftw_destroy_plan(q->dst2);
    fftw_free(q->re); fftw_free(q->im);
    fftw_free(q->kin_phase_re); fftw_free(q->kin_phase_im); fftw_free(q->kin_energy);
    fftw_free(q->scratch_re); fftw_free(q->scratch_im);
    memset(q, 0, sizeof(*q));
}

static bool quantum_alloc(QuantumField *q, int mx, int my) {
    memset(q, 0, sizeof(*q));
    q->mx = mx; q->my = my;
    size_t n = (size_t)mx * (size_t)my;
    q->re = fftw_alloc_real(n);
    q->im = fftw_alloc_real(n);
    q->kin_phase_re = fftw_alloc_real(n);
    q->kin_phase_im = fftw_alloc_real(n);
    q->kin_energy = fftw_alloc_real(n);
    q->scratch_re = fftw_alloc_real(n);
    q->scratch_im = fftw_alloc_real(n);
    if (!q->re || !q->im || !q->kin_phase_re || !q->kin_phase_im || !q->kin_energy ||
        !q->scratch_re || !q->scratch_im) {
        quantum_free(q);
        return false;
    }
    memset(q->re, 0, n * sizeof(double));
    memset(q->im, 0, n * sizeof(double));
    /* One in-place 2D DST-I plan, executed on re and im via the new-array interface
       (both arrays come from fftw_alloc_real, so alignment matches). */
    q->dst2 = fftw_plan_r2r_2d(my, mx, q->re, q->re, FFTW_RODFT00, FFTW_RODFT00, FFTW_MEASURE);
    if (!q->dst2) { quantum_free(q); return false; }
    memset(q->re, 0, n * sizeof(double));      /* FFTW_MEASURE scribbles on the array */
    return true;
}

static void quantum_build_kinetic(QuantumField *q) {
    for (int j = 0; j < q->my; ++j) {
        double ky = (double)(j + 1) * M_PI / q->Ly;
        for (int i = 0; i < q->mx; ++i) {
            double kx = (double)(i + 1) * M_PI / q->Lx;
            double T = 0.5 * q->hbar * q->hbar / q->mass * (kx*kx + ky*ky);
            double phase = -T * q->dt / q->hbar;
            size_t id = (size_t)j * (size_t)q->mx + (size_t)i;
            q->kin_energy[id] = T;
            q->kin_phase_re[id] = cos(phase);
            q->kin_phase_im[id] = sin(phase);
        }
    }
}

static double quantum_norm(const QuantumField *q) {
    double sum = 0.0;
    size_t n = (size_t)q->mx * (size_t)q->my;
    for (size_t k = 0; k < n; ++k)
        sum += q->re[k]*q->re[k] + q->im[k]*q->im[k];
    return sum * q->dx * q->dy;
}

static void quantum_normalize(QuantumField *q) {
    double sum = quantum_norm(q);
    if (sum <= 0.0) return;
    double s = 1.0 / sqrt(sum);
    size_t n = (size_t)q->mx * (size_t)q->my;
    for (size_t k = 0; k < n; ++k) { q->re[k] *= s; q->im[k] *= s; }
}

static void quantum_init_packet(QuantumField *q, const Params *p) {
    double theta = p->angle_deg * M_PI / 180.0;
    double x0 = 0.15 * p->box_w;
    double y0 = 0.20 * p->box_h;
    double sigma = 0.06 * fmin(p->box_w, p->box_h);
    double px = q->mass * p->speed * cos(theta);
    double py = -q->mass * p->speed * sin(theta);

    for (int j = 0; j < q->my; ++j) {
        double y = (double)(j + 1) * q->dy;
        for (int i = 0; i < q->mx; ++i) {
            double x = (double)(i + 1) * q->dx;
            double dx = x - x0, dy = y - y0;
            double env = exp(-(dx*dx + dy*dy)/(2.0*sigma*sigma));
            double ph = (px*x + py*y) / q->hbar;
            size_t id = (size_t)j * (size_t)q->mx + (size_t)i;
            q->re[id] = env * cos(ph);
            q->im[id] = env * sin(ph);
        }
    }
    quantum_normalize(q);
    q->steps = 0;
}

static void quantum_reset(QuantumField *q, const Params *p) {
    quantum_free(q);
    /* 20 points per unit length: the default packet (p = 28, wavelength 0.22)
       keeps ~4.5 points per wavelength, which also displays without moire */
    int mx = (int)CLAMP((int)(p->box_w * 20.0f), 120, 400);
    int my = (int)CLAMP((int)(p->box_h * 20.0f), 80, 320);
    if (!quantum_alloc(q, mx, my)) {
        fprintf(stderr, "Quantum allocation/plan creation failed.\n");
        exit(1);
    }
    q->Lx = p->box_w;
    q->Ly = p->box_h;
    q->dx = q->Lx / (double)(mx + 1);
    q->dy = q->Ly / (double)(my + 1);
    q->mass = 1.0;
    q->hbar = 1.0;
    q->dt = 0.0022;
    quantum_build_kinetic(q);
    quantum_init_packet(q, p);
}

/* exp(-i V dt/2 hbar) with V = -m g y  ->  multiply by exp(+i m g y dt / 2 hbar) */
static void quantum_half_potential_step(QuantumField *q, float gravity) {
    double coeff = 0.5 * q->dt * q->mass * gravity / q->hbar;
    for (int j = 0; j < q->my; ++j) {
        double ph = coeff * (double)(j + 1) * q->dy;
        double c = cos(ph), s = sin(ph);
        double *re = q->re + (size_t)j * (size_t)q->mx;
        double *im = q->im + (size_t)j * (size_t)q->mx;
        for (int i = 0; i < q->mx; ++i) {
            double a = re[i], b = im[i];
            re[i] = c*a - s*b;
            im[i] = s*a + c*b;
        }
    }
}

/* Forward 2D DST-I of re and im, phase rotation per mode, inverse transform.
   DST-I is its own inverse up to 2(N+1) per dimension. */
static void quantum_kinetic_step(QuantumField *q) {
    size_t n = (size_t)q->mx * (size_t)q->my;
    fftw_execute_r2r(q->dst2, q->re, q->re);
    fftw_execute_r2r(q->dst2, q->im, q->im);
    double scale = 1.0 / (4.0 * (double)(q->mx + 1) * (double)(q->my + 1));
    for (size_t k = 0; k < n; ++k) {
        double a = q->re[k], b = q->im[k];
        double c = q->kin_phase_re[k], s = q->kin_phase_im[k];
        q->re[k] = (c*a - s*b) * scale;
        q->im[k] = (s*a + c*b) * scale;
    }
    fftw_execute_r2r(q->dst2, q->re, q->re);
    fftw_execute_r2r(q->dst2, q->im, q->im);
}

/* <T> = sum T_nm |c_nm|^2 / sum |c_nm|^2 (Parseval), on a copy so psi is untouched. */
static double quantum_kinetic_energy(const QuantumField *q) {
    size_t n = (size_t)q->mx * (size_t)q->my;
    memcpy(q->scratch_re, q->re, n * sizeof(double));
    memcpy(q->scratch_im, q->im, n * sizeof(double));
    fftw_execute_r2r(q->dst2, q->scratch_re, q->scratch_re);
    fftw_execute_r2r(q->dst2, q->scratch_im, q->scratch_im);
    double tsum = 0.0, nsum = 0.0;
    for (size_t k = 0; k < n; ++k) {
        double w = q->scratch_re[k]*q->scratch_re[k] + q->scratch_im[k]*q->scratch_im[k];
        tsum += w * q->kin_energy[k];
        nsum += w;
    }
    return nsum > 0.0 ? tsum / nsum : 0.0;
}

/* No renormalisation: the split-step sine method is unitary, and the
   displayed norm is the diagnostic that shows it. */
static void quantum_step(QuantumField *q, float gravity, int substeps) {
    for (int s = 0; s < substeps; ++s) {
        quantum_half_potential_step(q, gravity);
        quantum_kinetic_step(q);
        quantum_half_potential_step(q, gravity);
        q->steps++;
    }
}

static void quantum_expectation(const QuantumField *q, double *xbar, double *ybar, double *norm) {
    double sx = 0.0, sy = 0.0, sn = 0.0;
    for (int j = 0; j < q->my; ++j) {
        double y = (double)(j + 1) * q->dy;
        for (int i = 0; i < q->mx; ++i) {
            double x = (double)(i + 1) * q->dx;
            size_t id = (size_t)j * (size_t)q->mx + (size_t)i;
            double rho = q->re[id]*q->re[id] + q->im[id]*q->im[id];
            sx += rho * x; sy += rho * y; sn += rho;
        }
    }
    *xbar = sn > 0.0 ? sx / sn : 0.0;
    *ybar = sn > 0.0 ? sy / sn : 0.0;
    *norm = sn * q->dx * q->dy;
}

/* <E> = <T> + <V>, with <V> = -m g <y> */
static double quantum_energy(const QuantumField *q, float gravity) {
    double xb, yb, nm;
    quantum_expectation(q, &xb, &yb, &nm);
    return quantum_kinetic_energy(q) - q->mass * gravity * yb;
}

/* ------------------------------ Eigenstates -------------------------------- */
/* 1D problem in y (hbar = m = 1):  -1/2 phi'' - g y phi = E phi,  phi(0) = phi(Ly) = 0.
   Numerov shooting from y = 0; the m-th state is found by bisection on E
   using the node count. */

static int numerov_shoot(double E, double g, double Ly, int N, double *phi_out, double *end_val) {
    double h = Ly / N, h2 = h * h;
    double prev = 0.0, cur = 1e-6, nodes = 0;
    if (phi_out) { phi_out[0] = prev; phi_out[1] = cur; }
    for (int k = 1; k < N; ++k) {
        double yk = k * h, ym = (k - 1) * h, yp = (k + 1) * h;
        double fk = 2.0 * (E + g * yk), fm = 2.0 * (E + g * ym), fp = 2.0 * (E + g * yp);
        double next = (2.0 * cur * (1.0 - 5.0 * h2 * fk / 12.0) - prev * (1.0 + h2 * fm / 12.0))
                      / (1.0 + h2 * fp / 12.0);
        if ((next < 0) != (cur < 0)) nodes++;
        prev = cur; cur = next;
        if (phi_out) phi_out[k + 1] = cur;
        if (fabs(cur) > 1e100) {          /* keep the integration in range */
            double sc = 1e-100;
            cur *= sc; prev *= sc;
            if (phi_out) for (int t = 0; t <= k + 1; ++t) phi_out[t] *= sc;
        }
    }
    *end_val = cur;
    return (int)nodes;
}

/* Eigenvalue number m (1-based), i.e. m-1 interior nodes. The node count is
   a staircase in E, so bisection on it brackets E_m; inside that bracket
   phi(Ly) changes sign exactly at E_m, so a second bisection on the sign of
   phi(Ly) pins it down. */
static double eigen_1d(int m, double g, double Ly, int N, double *phi) {
    double end;
    double Elo = -g * Ly - 1.0, Ehi = Elo, step = 1.0;
    while (numerov_shoot(Ehi, g, Ly, N, NULL, &end) < m) { Ehi += step; step *= 1.5; }
    while (Ehi - Elo > 1e-3) {
        double Em = 0.5 * (Elo + Ehi);
        if (numerov_shoot(Em, g, Ly, N, NULL, &end) < m) Elo = Em; else Ehi = Em;
    }
    double flo, fhi;
    numerov_shoot(Elo, g, Ly, N, NULL, &flo);
    numerov_shoot(Ehi, g, Ly, N, NULL, &fhi);
    for (int it = 0; it < 100 && (flo < 0) != (fhi < 0); ++it) {
        double Em = 0.5 * (Elo + Ehi), fm;
        numerov_shoot(Em, g, Ly, N, NULL, &fm);
        if ((fm < 0) == (flo < 0)) { Elo = Em; flo = fm; } else { Ehi = Em; fhi = fm; }
        if (Ehi - Elo < 1e-13 * fmax(1.0, fabs(Em))) break;
    }
    double E = 0.5 * (Elo + Ehi);
    numerov_shoot(E, g, Ly, N, phi, &end);
    return E;
}

/* Airy zeros a_k (Ai(a_k) = 0) for the bouncer energies E_k = |a_k| (g^2/2)^(1/3)
   above the floor, in the limit where the ceiling is far away. */
static const double airy_zeros[] = {
    2.33810741, 4.08794944, 5.52055983, 6.78670809, 7.94413359, 9.02265085,
    10.04017434, 11.00852430, 11.93601556, 12.82877675, 13.69148904, 14.52782995,
    15.34075514, 16.13268516, 16.90563400, 17.66130011, 18.40113260, 19.12638047,
    19.83812989, 20.53733291, 21.22482994, 21.90136760, 22.56761292, 23.22416501
};

static void eigen_free(EigenState *e) {
    free(e->phi);
    memset(e, 0, sizeof *e);
}

static void eigen_compute(EigenState *e, const QuantumField *q, float gravity, int n, int m) {
    eigen_free(e);
    e->n = n; e->m = m; e->my = q->my; e->Ly = q->Ly;
    int N = 4000;
    double *phi = malloc((size_t)(N + 1) * sizeof *phi);
    e->phi = malloc((size_t)q->my * sizeof *e->phi);
    if (!phi || !e->phi) { fprintf(stderr, "out of memory\n"); exit(1); }
    double Ey = eigen_1d(m, gravity, q->Ly, N, phi);
    /* sample on the interior grid and normalise */
    double norm = 0.0;
    for (int j = 0; j < q->my; ++j) {
        double y = (double)(j + 1) * q->dy;
        double pos = y / q->Ly * N;
        int k = (int)pos; if (k >= N) k = N - 1;
        double f = pos - k;
        double v = phi[k] * (1.0 - f) + phi[k + 1] * f;
        e->phi[j] = v;
        norm += v * v;
    }
    norm *= q->dy;
    double s = norm > 0 ? 1.0 / sqrt(norm) : 1.0;
    for (int j = 0; j < q->my; ++j) e->phi[j] *= s;
    free(phi);
    double kx = n * M_PI / q->Lx;
    double Ex = 0.5 * kx * kx;
    e->energy = Ex + Ey;
    e->energy_airy = Ex - gravity * q->Ly + airy_zeros[m - 1] * cbrt(0.5 * gravity * gravity);
    double amax = 0.0;
    for (int j = 0; j < q->my; ++j) amax = fmax(amax, fabs(e->phi[j]));
    e->nodes = 0;
    double last = 0.0;
    for (int j = 0; j < q->my - 2; ++j) {
        double v = e->phi[j];
        if (fabs(v) < 1e-3 * amax) continue;
        if (last != 0.0 && (v < 0) != (last < 0)) e->nodes++;
        last = v;
    }
}

/* Load psi_{n,m} into the field (real, so it can also be evolved: it should just
   acquire a phase). */
static void eigen_load(const EigenState *e, QuantumField *q) {
    double norm = 0.0;
    for (int j = 0; j < q->my; ++j) {
        for (int i = 0; i < q->mx; ++i) {
            double x = (double)(i + 1) * q->dx;
            double v = sin(e->n * M_PI * x / q->Lx) * e->phi[j];
            size_t id = (size_t)j * (size_t)q->mx + (size_t)i;
            q->re[id] = v; q->im[id] = 0.0;
            norm += v * v;
        }
    }
    (void)norm;
    quantum_normalize(q);
    q->steps = 0;
}

/* ------------------------------- Rendering -------------------------------- */

static void draw_circle(SDL_Renderer *r, int cx, int cy, int radius, SDL_Color col) {
    SDL_SetRenderDrawColor(r, col.r, col.g, col.b, col.a);
    for (int dy = -radius; dy <= radius; ++dy) {
        int dxlim = (int)sqrt((double)(radius*radius - dy*dy));
        SDL_RenderDrawLine(r, cx - dxlim, cy + dy, cx + dxlim, cy + dy);
    }
}

static SDL_Rect compute_sim_rect(const Params *p) {
    int avail_w = WINDOW_W - CONTROL_W - 2*MARGIN;
    int avail_h = WINDOW_H - 2*MARGIN;
    float scale = fminf((float)avail_w / p->box_w, (float)avail_h / p->box_h);
    SDL_Rect rc;
    rc.w = (int)(p->box_w * scale);
    rc.h = (int)(p->box_h * scale);
    rc.x = CONTROL_W + (avail_w - rc.w)/2 + MARGIN;
    rc.y = (avail_h - rc.h)/2 + MARGIN;
    return rc;
}

static void draw_box(SDL_Renderer *r, const SDL_Rect *sim) {
    SDL_SetRenderDrawColor(r, 220,220,220,255);
    SDL_RenderDrawRect(r, sim);
}

static void render_classical(SDL_Renderer *r, const SDL_Rect *sim, const Params *p, const ClassicalBall *b) {
    draw_box(r, sim);
    int sx = sim->x + (int)((b->x / p->box_w) * sim->w);
    int sy = sim->y + (int)((b->y / p->box_h) * sim->h);
    int sr = (int)(b->radius * (float)sim->w / p->box_w);
    if (sr < 3) sr = 3;
    draw_circle(r, sx, sy, sr, (SDL_Color){240,180,70,255});
}

/* |psi|^2 goes into a streaming texture the size of the grid, which the
   renderer scales to the box with linear filtering. */
static SDL_Texture *density_texture(SDL_Renderer *r, SDL_Texture *tex, const QuantumField *q) {
    int tw = 0, th = 0;
    if (tex) SDL_QueryTexture(tex, NULL, NULL, &tw, &th);
    if (!tex || tw != q->mx || th != q->my) {
        if (tex) SDL_DestroyTexture(tex);
        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
        tex = SDL_CreateTexture(r, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, q->mx, q->my);
        if (!tex) return NULL;
    }
    size_t n = (size_t)q->mx * (size_t)q->my;
    double maxrho = 1e-300;
    for (size_t k = 0; k < n; ++k) {
        double rho = q->re[k]*q->re[k] + q->im[k]*q->im[k];
        if (rho > maxrho) maxrho = rho;
    }
    void *pixels; int pitch;
    if (SDL_LockTexture(tex, NULL, &pixels, &pitch) != 0) return tex;
    for (int j = 0; j < q->my; ++j) {
        Uint32 *row = (Uint32 *)((Uint8 *)pixels + (size_t)j * (size_t)pitch);
        for (int i = 0; i < q->mx; ++i) {
            size_t id = (size_t)j * (size_t)q->mx + (size_t)i;
            double rho = q->re[id]*q->re[id] + q->im[id]*q->im[id];
            double t = sqrt(rho / maxrho);            /* sqrt: shows the faint tails */
            Uint32 cr = (Uint32)(20 + 20 * t);
            Uint32 cg = (Uint32)(20 + 215 * t);
            Uint32 cb = (Uint32)(28 + 210 * t);
            row[i] = 0xFF000000u | (cr << 16) | (cg << 8) | cb;
        }
    }
    SDL_UnlockTexture(tex);
    return tex;
}

static void render_quantum(SDL_Renderer *r, const SDL_Rect *sim, const QuantumField *q, SDL_Texture *tex, bool show_mean) {
    /* the texture covers the interior points; stretch it so that the walls
       (where psi = 0) fall half a cell outside the first/last texel */
    SDL_Rect dst = *sim;
    float cw = (float)sim->w / (float)(q->mx + 1), ch = (float)sim->h / (float)(q->my + 1);
    dst.x += (int)(0.5f * cw); dst.y += (int)(0.5f * ch);
    dst.w -= (int)cw; dst.h -= (int)ch;
    if (tex) SDL_RenderCopy(r, tex, NULL, &dst);
    if (show_mean) {
        double xbar, ybar, nm;
        quantum_expectation(q, &xbar, &ybar, &nm);
        int sx = sim->x + (int)((xbar / q->Lx) * sim->w);
        int sy = sim->y + (int)((ybar / q->Ly) * sim->h);
        draw_circle(r, sx, sy, 4, (SDL_Color){255,255,255,255});
    }
    draw_box(r, sim);
}

/* ------------------------------ Self test ---------------------------------- */
/* Runs the quantum solver without a window: checks that the norm is conserved
   and that the energy is conserved, and that a computed eigenstate is
   stationary. Exit status 0 on success. */
static int self_test(void) {
    Params p = { .angle_deg = 55.0f, .speed = 28.0f, .gravity = 18.0f, .box_w = 12.0f, .box_h = 8.0f };
    QuantumField q; memset(&q, 0, sizeof q);
    quantum_reset(&q, &p);
    double n0 = quantum_norm(&q);
    double e0 = quantum_energy(&q, p.gravity);
    quantum_step(&q, p.gravity, 3000);
    double n1 = quantum_norm(&q), e1 = quantum_energy(&q, p.gravity);
    double dn = fabs(n1 - n0), de = fabs(e1 - e0) / fabs(e0);
    printf("packet: 3000 steps  norm %.7f -> %.7f (drift %.1e)  energy %.5f -> %.5f (rel drift %.1e)\n",
           n0, n1, dn, e0, e1, de);
    int ok = dn < 1e-9 && de < 1e-4;

    EigenState e; memset(&e, 0, sizeof e);
    for (int m = 1; m <= 3; ++m) {
        eigen_compute(&e, &q, p.gravity, 1, m);
        eigen_load(&e, &q);
        double xb0, yb0, nm0; quantum_expectation(&q, &xb0, &yb0, &nm0);
        quantum_step(&q, p.gravity, 500);
        double xb1, yb1, nm1; quantum_expectation(&q, &xb1, &yb1, &nm1);
        double eq = quantum_energy(&q, p.gravity);
        double dy = fabs(yb1 - yb0);
        printf("eigen n=1 m=%d: E %.5f (airy %.5f), <E> after 500 steps %.5f, <y> moved %.1e, nodes %d\n",
               m, e.energy, e.energy_airy, eq, dy, e.nodes);
        ok = ok && dy < 1e-3 && fabs(eq - e.energy) < 2e-3 * fabs(e.energy) && e.nodes == m - 1;
    }
    eigen_free(&e);
    quantum_free(&q);
    printf("%s\n", ok ? "self-test passed" : "SELF-TEST FAILED");
    return ok ? 0 : 1;
}

static int write_ppm(SDL_Renderer *r, const char *path) {
    int w = WINDOW_W, h = WINDOW_H;
    unsigned char *pix = malloc((size_t)w * h * 3);
    if (!pix) return -1;
    if (SDL_RenderReadPixels(r, NULL, SDL_PIXELFORMAT_RGB24, pix, w * 3) != 0) { free(pix); return -1; }
    FILE *f = fopen(path, "wb");
    if (!f) { free(pix); return -1; }
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    fwrite(pix, 1, (size_t)w * h * 3, f);
    fclose(f);
    free(pix);
    return 0;
}

static void usage(FILE *f) {
    fprintf(f,
        "usage: quantum_ball [--mode classical|quantum|eigen] [--state N,M] [--frames N] [--record DIR]\n"
        "       quantum_ball --selftest | --version | -h\n"
        "\n"
        "  --mode M      start in the given mode (default classical)\n"
        "  --state N,M   start in eigenstate mode showing psi_{N,M}\n"
        "  --frames N    run N frames and exit (for recording and tests)\n"
        "  --record DIR  write every frame to DIR/frame_NNNNN.ppm\n"
        "  --selftest    check norm/energy conservation and eigenstates, no window\n");
}

/* --------------------------------- Main ----------------------------------- */

int main(int argc, char **argv) {
    SimMode mode = MODE_CLASSICAL;
    long max_frames = -1;
    const char *record_dir = NULL;
    int start_n = 1, start_m = 1;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(stdout); return 0; }
        if (!strcmp(argv[i], "--version")) { printf("quantum_ball %s\n", VERSION); return 0; }
        if (!strcmp(argv[i], "--selftest")) return self_test();
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "classical")) mode = MODE_CLASSICAL;
            else if (!strcmp(m, "quantum")) mode = MODE_QUANTUM;
            else if (!strcmp(m, "eigen")) mode = MODE_EIGEN;
            else { usage(stderr); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "--frames") && i + 1 < argc) { max_frames = atol(argv[++i]); continue; }
        if (!strcmp(argv[i], "--state") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%d", &start_n, &start_m) != 2 ||
                start_n < 1 || start_n > MAX_EIGEN_N || start_m < 1 || start_m > MAX_EIGEN_M) { usage(stderr); return 2; }
            mode = MODE_EIGEN;
            continue;
        }
        if (!strcmp(argv[i], "--record") && i + 1 < argc) { record_dir = argv[++i]; continue; }
        usage(stderr);
        return 2;
    }

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window *window = SDL_CreateWindow(
        "Quantum Bouncing Ball",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    Uint32 rflags = record_dir ? SDL_RENDERER_SOFTWARE : (SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, rflags);
    if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    Params p = { .angle_deg = 55.0f, .speed = 28.0f, .gravity = 18.0f, .box_w = 12.0f, .box_h = 8.0f };
    ClassicalBall cball;
    QuantumField qfield; memset(&qfield, 0, sizeof qfield);
    EigenState eig; memset(&eig, 0, sizeof eig);
    int eig_n = start_n, eig_m = start_m;
    classical_reset(&cball, &p);
    quantum_reset(&qfield, &p);
    double e_ref = 0.0;            /* energy at reset, for the drift display */
    bool have_e_ref = false;

    Slider sliders[5];
    int sx = 30, sy = 80, sw = CONTROL_W - 60, sh = 24, gap = 80;
    sliders[0] = (Slider){ .rect = {sx, sy + 0*gap, sw, sh}, .minv = 5.0f, .maxv = 85.0f, .value = &p.angle_deg, .label = "ANGLE" };
    sliders[1] = (Slider){ .rect = {sx, sy + 1*gap, sw, sh}, .minv = 2.0f, .maxv = 60.0f, .value = &p.speed, .label = "SPEED" };
    sliders[2] = (Slider){ .rect = {sx, sy + 2*gap, sw, sh}, .minv = 0.0f, .maxv = 45.0f, .value = &p.gravity, .label = "GRAVITY" };
    sliders[3] = (Slider){ .rect = {sx, sy + 3*gap, sw, sh}, .minv = 6.0f, .maxv = 20.0f, .value = &p.box_w, .label = "BOX W" };
    sliders[4] = (Slider){ .rect = {sx, sy + 4*gap, sw, sh}, .minv = 4.0f, .maxv = 14.0f, .value = &p.box_h, .label = "BOX H" };

    Button btn_classical = {{30, 480, 120, 36}, "CLASSICAL"};
    Button btn_quantum   = {{170, 480, 120, 36}, "QUANTUM"};
    Button btn_eigen     = {{30, 526, 260, 36}, "EIGENSTATES (N,M)"};
    Button btn_reset     = {{30, 572, 120, 36}, "RESET"};
    Button btn_pause     = {{170, 572, 120, 36}, "PAUSE"};

    SDL_Texture *tex = NULL;
    bool paused = false, running = true, need_reset = true;
    long frame = 0;
    Uint64 prev = SDL_GetPerformanceCounter();
    double freq = (double)SDL_GetPerformanceFrequency();

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                int mx = e.button.x, my = e.button.y;
                for (int i = 0; i < 5; ++i) {
                    if (point_in_rect(mx, my, sliders[i].rect)) {
                        sliders[i].dragging = true;
                        slider_handle_mouse(&sliders[i], mx);
                        need_reset = true;
                    }
                }
                if (point_in_rect(mx, my, btn_classical.rect)) { mode = MODE_CLASSICAL; need_reset = true; }
                if (point_in_rect(mx, my, btn_quantum.rect))   { mode = MODE_QUANTUM;   need_reset = true; }
                if (point_in_rect(mx, my, btn_eigen.rect))     { mode = MODE_EIGEN;     need_reset = true; }
                if (point_in_rect(mx, my, btn_reset.rect))     { need_reset = true; }
                if (point_in_rect(mx, my, btn_pause.rect))     { paused = !paused; }
            } else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                for (int i = 0; i < 5; ++i) sliders[i].dragging = false;
            } else if (e.type == SDL_MOUSEMOTION) {
                for (int i = 0; i < 5; ++i)
                    if (sliders[i].dragging) { slider_handle_mouse(&sliders[i], e.motion.x); need_reset = true; }
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE: running = false; break;
                    case SDLK_SPACE: paused = !paused; break;
                    case SDLK_r: need_reset = true; break;
                    case SDLK_c: mode = MODE_CLASSICAL; need_reset = true; break;
                    case SDLK_q: mode = MODE_QUANTUM; need_reset = true; break;
                    case SDLK_e: mode = MODE_EIGEN; need_reset = true; break;
                    case SDLK_UP:    if (mode == MODE_EIGEN && eig_m < MAX_EIGEN_M) { eig_m++; need_reset = true; } break;
                    case SDLK_DOWN:  if (mode == MODE_EIGEN && eig_m > 1) { eig_m--; need_reset = true; } break;
                    case SDLK_RIGHT: if (mode == MODE_EIGEN && eig_n < MAX_EIGEN_N) { eig_n++; need_reset = true; } break;
                    case SDLK_LEFT:  if (mode == MODE_EIGEN && eig_n > 1) { eig_n--; need_reset = true; } break;
                    default: break;
                }
            }
        }

        if (need_reset) {
            need_reset = false;
            classical_reset(&cball, &p);
            quantum_reset(&qfield, &p);
            if (mode == MODE_EIGEN) {
                eigen_compute(&eig, &qfield, p.gravity, eig_n, eig_m);
                eigen_load(&eig, &qfield);
            }
            have_e_ref = false;
        }

        Uint64 now = SDL_GetPerformanceCounter();
        float dt_wall = (float)((double)(now - prev) / freq);
        prev = now;
        dt_wall = CLAMP(dt_wall, 0.0f, 0.03f);
        if (record_dir) dt_wall = 1.0f / 60.0f;      /* deterministic when recording */

        if (!paused) {
            if (mode == MODE_CLASSICAL) classical_step(&cball, &p, dt_wall);
            else quantum_step(&qfield, p.gravity, 3);
        }

        /* ---- draw ---- */
        SDL_SetRenderDrawColor(renderer, 20,20,28,255);
        SDL_RenderClear(renderer);
        SDL_Rect control = {0,0,CONTROL_W,WINDOW_H};
        SDL_SetRenderDrawColor(renderer, 35,35,45,255);
        SDL_RenderFillRect(renderer, &control);
        SDL_SetRenderDrawColor(renderer, 80,80,90,255);
        SDL_RenderDrawLine(renderer, CONTROL_W, 0, CONTROL_W, WINDOW_H);

        SDL_Color white = {245,245,245,255}, dim = {180,180,200,255}, cyan = {170,220,230,255}, amber = {240,200,120,255};
        draw_text(renderer, 28, 20, 3, "QUANTUM BOUNCER", white);
        for (int i = 0; i < 5; ++i) draw_slider(renderer, &sliders[i]);
        draw_button(renderer, &btn_classical, mode == MODE_CLASSICAL);
        draw_button(renderer, &btn_quantum, mode == MODE_QUANTUM);
        draw_button(renderer, &btn_eigen, mode == MODE_EIGEN);
        draw_button(renderer, &btn_reset, false);
        draw_button(renderer, &btn_pause, paused);

        char line[128];
        int ty = 626;
        if (mode == MODE_CLASSICAL) {
            double E = classical_energy(&cball, &p);
            if (!have_e_ref) { e_ref = E; have_e_ref = true; }
            draw_text(renderer, 30, ty, 2, "POINT MASS, ELASTIC WALLS", cyan); ty += 24;
            snprintf(line, sizeof line, "E %.4f", E);           draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
            snprintf(line, sizeof line, "DRIFT %.1e", E - e_ref); draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
        } else {
            double xb, yb, norm;
            quantum_expectation(&qfield, &xb, &yb, &norm);
            double E = quantum_kinetic_energy(&qfield) - qfield.mass * p.gravity * yb;
            if (!have_e_ref) { e_ref = E; have_e_ref = true; }
            if (mode == MODE_QUANTUM) {
                draw_text(renderer, 30, ty, 2, "STRANG SPLIT STEP, DST-I", cyan); ty += 24;
            } else {
                snprintf(line, sizeof line, "STATE N=%d M=%d  ARROWS", eig.n, eig.m);
                draw_text(renderer, 30, ty, 2, line, cyan); ty += 24;
                snprintf(line, sizeof line, "E(N,M) %.4f", eig.energy);      draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
                snprintf(line, sizeof line, "AIRY   %.4f", eig.energy_airy); draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
            }
            snprintf(line, sizeof line, "<E> %.4f", E);                       draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
            snprintf(line, sizeof line, "DRIFT %.1e", have_e_ref ? E - e_ref : 0.0); draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
            snprintf(line, sizeof line, "NORM %.6f", norm);                   draw_text(renderer, 30, ty, 2, line, amber); ty += 24;
            snprintf(line, sizeof line, "GRID %dX%d DT %.4f", qfield.mx, qfield.my, qfield.dt); draw_text(renderer, 30, ty, 2, line, dim); ty += 24;
            snprintf(line, sizeof line, "STEPS %ld", qfield.steps);           draw_text(renderer, 30, ty, 2, line, dim); ty += 24;
        }
        ty += 8;
        draw_text(renderer, 30, ty, 2, "KEYS C Q E R SPACE ESC", dim); ty += 24;
        if (mode != MODE_CLASSICAL) { draw_text(renderer, 30, ty, 2, "WHITE DOT = <X>,<Y>", dim); ty += 24; }
        draw_text(renderer, 30, ty, 2, "GRAVITY POINTS DOWN", dim);

        SDL_Rect sim = compute_sim_rect(&p);
        if (mode == MODE_CLASSICAL) render_classical(renderer, &sim, &p, &cball);
        else { tex = density_texture(renderer, tex, &qfield); render_quantum(renderer, &sim, &qfield, tex, true); }

        SDL_RenderPresent(renderer);

        if (record_dir) {
            char path[1024];
            snprintf(path, sizeof path, "%s/frame_%05ld.ppm", record_dir, frame);
            if (write_ppm(renderer, path) != 0) { fprintf(stderr, "cannot write %s\n", path); running = false; }
        }
        frame++;
        if (max_frames >= 0 && frame >= max_frames) running = false;
    }

    if (tex) SDL_DestroyTexture(tex);
    eigen_free(&eig);
    quantum_free(&qfield);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
