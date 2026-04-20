#include <SDL2/SDL.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
    quantum_ball_sdl2.c

    A self-contained SDL2 demo that simulates either:
      1) a classical ballistic ball in a rectangular box with elastic wall bounces, or
      2) a quantum Gaussian wave packet in a 2D box under gravity.

    UI:
      - Sliders for launch angle, initial speed, gravity, box width, box height
      - Toggle buttons for Classical / Quantum mode
      - Reset button
      - Pause button

    Notes on the quantum model:
      - This solves the time-dependent Schrödinger equation on a rectangular grid:
            i ħ dψ/dt = [-(ħ² / 2m) ∇² + V(x,y)] ψ
        with V(x,y) = m g y inside the box and hard-wall boundary conditions.
      - Time stepping uses RK4 on the real/imaginary fields.
      - This is a compact demo, not a high-order or unconditionally stable PDE code.
      - Units are normalized / demonstrational rather than tied to SI.

    Build on Linux:
      gcc -O2 -std=c11 quantum_ball_sdl2.c -o quantum_ball_sdl2 -lSDL2 -lm

    On macOS with Homebrew SDL2 installed:
      clang -O2 -std=c11 quantum_ball_sdl2.c -o quantum_ball_sdl2 $(sdl2-config --cflags --libs) -lm
*/

#define WINDOW_W 1400
#define WINDOW_H 900

#define CONTROL_W 320
#define MARGIN 16

#define MAX_GRID_W 220
#define MAX_GRID_H 220

#define CLAMP(x,a,b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))
#define IDX(i,j,nx) ((j)*(nx) + (i))

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum {
    MODE_CLASSICAL = 0,
    MODE_QUANTUM = 1
} SimMode;

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
    float minv;
    float maxv;
    float *value;
    const char *label;
    bool dragging;
} Slider;

typedef struct {
    SDL_Rect rect;
    const char *label;
    bool pressed;
} Button;

typedef struct {
    int nx, ny;
    float dx, dy;
    float dt;
    float mass;
    float hbar;

    float *re;
    float *im;
    float *V;

    float *k1re, *k1im;
    float *k2re, *k2im;
    float *k3re, *k3im;
    float *k4re, *k4im;
    float *tmpre, *tmpim;
} QuantumField;

static SDL_Window *g_window = NULL;
static SDL_Renderer *g_renderer = NULL;

/* ----------------------------- Tiny bitmap font -----------------------------
   5x7 uppercase-ish font for the minimal labels in the control panel.
   Supported chars: A-Z, 0-9, space, colon, period, minus.
*/

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
    {' ',{0x00,0x00,0x00,0x00,0x00,0x00,0x00}},
};

static const Glyph *find_glyph(char c) {
    size_t n = sizeof(g_font) / sizeof(g_font[0]);
    if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    for (size_t i = 0; i < n; ++i) {
        if (g_font[i].c == c) return &g_font[i];
    }
    return &g_font[sizeof(g_font) / sizeof(g_font[0]) - 1];
}

static void draw_char(SDL_Renderer *r, int x, int y, int s, char c, SDL_Color col) {
    const Glyph *g = find_glyph(c);
    SDL_SetRenderDrawColor(r, col.r, col.g, col.b, col.a);
    for (int row = 0; row < 7; ++row) {
        unsigned char bits = g->rows[row];
        for (int colb = 0; colb < 5; ++colb) {
            if (bits & (1 << (4 - colb))) {
                SDL_Rect px = {x + colb * s, y + row * s, s, s};
                SDL_RenderFillRect(r, &px);
            }
        }
    }
}

static void draw_text(SDL_Renderer *r, int x, int y, int s, const char *text, SDL_Color col) {
    int cx = x;
    for (const char *p = text; *p; ++p) {
        draw_char(r, cx, y, s, *p, col);
        cx += 6 * s;
    }
}

static void format_float(char *buf, size_t n, const char *label, float v) {
    snprintf(buf, n, "%s %.2f", label, v);
}

/* ------------------------------ UI utilities ------------------------------- */

static bool point_in_rect(int x, int y, SDL_Rect rc) {
    return x >= rc.x && x < rc.x + rc.w && y >= rc.y && y < rc.y + rc.h;
}

static void draw_slider(SDL_Renderer *r, const Slider *s) {
    SDL_Color fg = {235, 235, 235, 255};
    SDL_Color track = {90, 90, 100, 255};
    SDL_Color knob = {220, 180, 70, 255};

    char buf[128];
    format_float(buf, sizeof(buf), s->label, *s->value);
    draw_text(r, s->rect.x, s->rect.y - 22, 2, buf, fg);

    SDL_Rect bar = {s->rect.x, s->rect.y + s->rect.h / 2 - 3, s->rect.w, 6};
    SDL_SetRenderDrawColor(r, track.r, track.g, track.b, track.a);
    SDL_RenderFillRect(r, &bar);

    float t = (*s->value - s->minv) / (s->maxv - s->minv);
    t = CLAMP(t, 0.0f, 1.0f);
    int knob_x = s->rect.x + (int)(t * s->rect.w);
    SDL_Rect k = {knob_x - 7, s->rect.y, 14, s->rect.h};
    SDL_SetRenderDrawColor(r, knob.r, knob.g, knob.b, knob.a);
    SDL_RenderFillRect(r, &k);
}

static void slider_handle_mouse(Slider *s, int mx) {
    float t = (float)(mx - s->rect.x) / (float)s->rect.w;
    t = CLAMP(t, 0.0f, 1.0f);
    *s->value = s->minv + t * (s->maxv - s->minv);
}

static void draw_button(SDL_Renderer *r, const Button *b, bool active) {
    SDL_Color edge = {180, 180, 180, 255};
    SDL_Color fill = active ? (SDL_Color){70, 120, 70, 255} : (SDL_Color){70, 70, 85, 255};
    SDL_Color fg = {240, 240, 240, 255};

    SDL_SetRenderDrawColor(r, fill.r, fill.g, fill.b, fill.a);
    SDL_RenderFillRect(r, &b->rect);
    SDL_SetRenderDrawColor(r, edge.r, edge.g, edge.b, edge.a);
    SDL_RenderDrawRect(r, &b->rect);
    draw_text(r, b->rect.x + 10, b->rect.y + 8, 2, b->label, fg);
}

/* --------------------------- Classical simulation -------------------------- */

static void classical_reset(ClassicalBall *b, const Params *p) {
    float theta = p->angle_deg * (float)M_PI / 180.0f;
    b->radius = 0.03f * (p->box_w < p->box_h ? p->box_w : p->box_h);
    b->x = 0.15f * p->box_w;
    b->y = 0.20f * p->box_h;
    b->vx = p->speed * cosf(theta);
    b->vy = -p->speed * sinf(theta);
}

static void classical_step(ClassicalBall *b, const Params *p, float dt) {
    b->vy += p->gravity * dt;
    b->x += b->vx * dt;
    b->y += b->vy * dt;

    if (b->x - b->radius < 0.0f) {
        b->x = b->radius;
        b->vx = -b->vx;
    }
    if (b->x + b->radius > p->box_w) {
        b->x = p->box_w - b->radius;
        b->vx = -b->vx;
    }
    if (b->y - b->radius < 0.0f) {
        b->y = b->radius;
        b->vy = -b->vy;
    }
    if (b->y + b->radius > p->box_h) {
        b->y = p->box_h - b->radius;
        b->vy = -b->vy;
    }
}

/* ---------------------------- Quantum simulation --------------------------- */

static void quantum_free(QuantumField *q) {
    free(q->re);   free(q->im);   free(q->V);
    free(q->k1re); free(q->k1im); free(q->k2re); free(q->k2im);
    free(q->k3re); free(q->k3im); free(q->k4re); free(q->k4im);
    free(q->tmpre); free(q->tmpim);
    memset(q, 0, sizeof(*q));
}

static bool quantum_alloc(QuantumField *q, int nx, int ny) {
    size_t n = (size_t)nx * (size_t)ny;
    memset(q, 0, sizeof(*q));
    q->nx = nx;
    q->ny = ny;

    q->re = (float*)calloc(n, sizeof(float));
    q->im = (float*)calloc(n, sizeof(float));
    q->V  = (float*)calloc(n, sizeof(float));

    q->k1re = (float*)calloc(n, sizeof(float)); q->k1im = (float*)calloc(n, sizeof(float));
    q->k2re = (float*)calloc(n, sizeof(float)); q->k2im = (float*)calloc(n, sizeof(float));
    q->k3re = (float*)calloc(n, sizeof(float)); q->k3im = (float*)calloc(n, sizeof(float));
    q->k4re = (float*)calloc(n, sizeof(float)); q->k4im = (float*)calloc(n, sizeof(float));
    q->tmpre = (float*)calloc(n, sizeof(float)); q->tmpim = (float*)calloc(n, sizeof(float));

    if (!q->re || !q->im || !q->V ||
        !q->k1re || !q->k1im || !q->k2re || !q->k2im ||
        !q->k3re || !q->k3im || !q->k4re || !q->k4im ||
        !q->tmpre || !q->tmpim) {
        quantum_free(q);
        return false;
    }
    return true;
}

static void quantum_fill_potential(QuantumField *q, const Params *p) {
    q->dx = p->box_w / (float)(q->nx - 1);
    q->dy = p->box_h / (float)(q->ny - 1);

    for (int j = 0; j < q->ny; ++j) {
        float y = j * q->dy;
        for (int i = 0; i < q->nx; ++i) {
            q->V[IDX(i,j,q->nx)] = q->mass * p->gravity * y;
        }
    }
}

static void quantum_normalize(QuantumField *q) {
    double sum = 0.0;
    for (int j = 0; j < q->ny; ++j) {
        for (int i = 0; i < q->nx; ++i) {
            int id = IDX(i,j,q->nx);
            double a2 = (double)q->re[id] * q->re[id] + (double)q->im[id] * q->im[id];
            sum += a2;
        }
    }
    sum *= (double)q->dx * (double)q->dy;
    if (sum <= 0.0) return;
    float inv = (float)(1.0 / sqrt(sum));
    size_t n = (size_t)q->nx * (size_t)q->ny;
    for (size_t k = 0; k < n; ++k) {
        q->re[k] *= inv;
        q->im[k] *= inv;
    }
}

static void quantum_init_packet(QuantumField *q, const Params *p) {
    float theta = p->angle_deg * (float)M_PI / 180.0f;
    float x0 = 0.15f * p->box_w;
    float y0 = 0.20f * p->box_h;

    float sigma = 0.06f * (p->box_w < p->box_h ? p->box_w : p->box_h);
    float px = q->mass * p->speed * cosf(theta);
    float py = -q->mass * p->speed * sinf(theta);

    for (int j = 0; j < q->ny; ++j) {
        float y = j * q->dy;
        for (int i = 0; i < q->nx; ++i) {
            float x = i * q->dx;
            float dx = x - x0;
            float dy = y - y0;
            float env = expf(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
            float phase = (px * x + py * y) / q->hbar;
            int id = IDX(i,j,q->nx);
            q->re[id] = env * cosf(phase);
            q->im[id] = env * sinf(phase);
        }
    }

    for (int i = 0; i < q->nx; ++i) {
        q->re[IDX(i,0,q->nx)] = q->im[IDX(i,0,q->nx)] = 0.0f;
        q->re[IDX(i,q->ny-1,q->nx)] = q->im[IDX(i,q->ny-1,q->nx)] = 0.0f;
    }
    for (int j = 0; j < q->ny; ++j) {
        q->re[IDX(0,j,q->nx)] = q->im[IDX(0,j,q->nx)] = 0.0f;
        q->re[IDX(q->nx-1,j,q->nx)] = q->im[IDX(q->nx-1,j,q->nx)] = 0.0f;
    }

    quantum_normalize(q);
}

static void quantum_reset(QuantumField *q, const Params *p) {
    quantum_free(q);

    int nx = (int)CLAMP(p->box_w * 0.85f, 80.0f, (float)MAX_GRID_W);
    int ny = (int)CLAMP(p->box_h * 0.85f, 80.0f, (float)MAX_GRID_H);

    if (!quantum_alloc(q, nx, ny)) {
        fprintf(stderr, "Allocation failed for quantum field.\n");
        exit(1);
    }

    q->mass = 1.0f;
    q->hbar = 1.0f;
    q->dt = 0.0008f;

    quantum_fill_potential(q, p);
    quantum_init_packet(q, p);
}

static void quantum_rhs(const QuantumField *q,
                        const float *re, const float *im,
                        float *dre, float *dim) {
    float inv_dx2 = 1.0f / (q->dx * q->dx);
    float inv_dy2 = 1.0f / (q->dy * q->dy);
    float kin = q->hbar / (2.0f * q->mass);

    for (int j = 0; j < q->ny; ++j) {
        for (int i = 0; i < q->nx; ++i) {
            int id = IDX(i,j,q->nx);
            if (i == 0 || i == q->nx - 1 || j == 0 || j == q->ny - 1) {
                dre[id] = 0.0f;
                dim[id] = 0.0f;
                continue;
            }

            int il = IDX(i-1,j,q->nx), ir = IDX(i+1,j,q->nx);
            int iu = IDX(i,j-1,q->nx), idn = IDX(i,j+1,q->nx);

            float lap_re = (re[il] - 2.0f*re[id] + re[ir]) * inv_dx2
                         + (re[iu] - 2.0f*re[id] + re[idn]) * inv_dy2;
            float lap_im = (im[il] - 2.0f*im[id] + im[ir]) * inv_dx2
                         + (im[iu] - 2.0f*im[id] + im[idn]) * inv_dy2;

            /*
               From i ħ ψ_t = -(ħ²/2m) Δψ + V ψ
               with ψ = a + i b:
                  a_t = -(ħ/2m) Δb + (V/ħ) b
                  b_t = +(ħ/2m) Δa - (V/ħ) a
            */
            float v_over_h = q->V[id] / q->hbar;
            dre[id] = -kin * lap_im + v_over_h * im[id];
            dim[id] =  kin * lap_re - v_over_h * re[id];
        }
    }
}

static void quantum_apply_bc(QuantumField *q, float *re, float *im) {
    for (int i = 0; i < q->nx; ++i) {
        re[IDX(i,0,q->nx)] = 0.0f;
        im[IDX(i,0,q->nx)] = 0.0f;
        re[IDX(i,q->ny-1,q->nx)] = 0.0f;
        im[IDX(i,q->ny-1,q->nx)] = 0.0f;
    }
    for (int j = 0; j < q->ny; ++j) {
        re[IDX(0,j,q->nx)] = 0.0f;
        im[IDX(0,j,q->nx)] = 0.0f;
        re[IDX(q->nx-1,j,q->nx)] = 0.0f;
        im[IDX(q->nx-1,j,q->nx)] = 0.0f;
    }
}

static void quantum_step(QuantumField *q, int substeps) {
    size_t n = (size_t)q->nx * (size_t)q->ny;
    for (int step = 0; step < substeps; ++step) {
        quantum_rhs(q, q->re, q->im, q->k1re, q->k1im);

        for (size_t k = 0; k < n; ++k) {
            q->tmpre[k] = q->re[k] + 0.5f * q->dt * q->k1re[k];
            q->tmpim[k] = q->im[k] + 0.5f * q->dt * q->k1im[k];
        }
        quantum_apply_bc(q, q->tmpre, q->tmpim);
        quantum_rhs(q, q->tmpre, q->tmpim, q->k2re, q->k2im);

        for (size_t k = 0; k < n; ++k) {
            q->tmpre[k] = q->re[k] + 0.5f * q->dt * q->k2re[k];
            q->tmpim[k] = q->im[k] + 0.5f * q->dt * q->k2im[k];
        }
        quantum_apply_bc(q, q->tmpre, q->tmpim);
        quantum_rhs(q, q->tmpre, q->tmpim, q->k3re, q->k3im);

        for (size_t k = 0; k < n; ++k) {
            q->tmpre[k] = q->re[k] + q->dt * q->k3re[k];
            q->tmpim[k] = q->im[k] + q->dt * q->k3im[k];
        }
        quantum_apply_bc(q, q->tmpre, q->tmpim);
        quantum_rhs(q, q->tmpre, q->tmpim, q->k4re, q->k4im);

        for (size_t k = 0; k < n; ++k) {
            q->re[k] += (q->dt / 6.0f) * (q->k1re[k] + 2.0f*q->k2re[k] + 2.0f*q->k3re[k] + q->k4re[k]);
            q->im[k] += (q->dt / 6.0f) * (q->k1im[k] + 2.0f*q->k2im[k] + 2.0f*q->k3im[k] + q->k4im[k]);
        }
        quantum_apply_bc(q, q->re, q->im);
        quantum_normalize(q);
    }
}

static void quantum_expectation(const QuantumField *q, float *xbar, float *ybar) {
    double sx = 0.0, sy = 0.0, norm = 0.0;
    for (int j = 0; j < q->ny; ++j) {
        float y = j * q->dy;
        for (int i = 0; i < q->nx; ++i) {
            float x = i * q->dx;
            int id = IDX(i,j,q->nx);
            double rho = (double)q->re[id]*q->re[id] + (double)q->im[id]*q->im[id];
            sx += rho * x;
            sy += rho * y;
            norm += rho;
        }
    }
    if (norm > 0.0) {
        *xbar = (float)(sx / norm);
        *ybar = (float)(sy / norm);
    } else {
        *xbar = *ybar = 0.0f;
    }
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
    int avail_w = WINDOW_W - CONTROL_W - 2 * MARGIN;
    int avail_h = WINDOW_H - 2 * MARGIN;

    float sx = (float)avail_w / p->box_w;
    float sy = (float)avail_h / p->box_h;
    float scale = sx < sy ? sx : sy;

    int w = (int)(p->box_w * scale);
    int h = (int)(p->box_h * scale);

    SDL_Rect rc;
    rc.x = CONTROL_W + (avail_w - w) / 2 + MARGIN;
    rc.y = (avail_h - h) / 2 + MARGIN;
    rc.w = w;
    rc.h = h;
    return rc;
}

static void world_to_screen(const SDL_Rect *sim, const Params *p, float x, float y, int *sx, int *sy) {
    float fx = x / p->box_w;
    float fy = y / p->box_h;
    *sx = sim->x + (int)(fx * sim->w);
    *sy = sim->y + (int)(fy * sim->h);
}

static void draw_box(SDL_Renderer *r, const SDL_Rect *sim) {
    SDL_SetRenderDrawColor(r, 220, 220, 220, 255);
    SDL_RenderDrawRect(r, sim);
}

static void render_classical(SDL_Renderer *r, const SDL_Rect *sim, const Params *p, const ClassicalBall *b) {
    draw_box(r, sim);

    int sx, sy;
    world_to_screen(sim, p, b->x, b->y, &sx, &sy);
    int sr = (int)(b->radius * (float)sim->w / p->box_w);
    if (sr < 3) sr = 3;
    draw_circle(r, sx, sy, sr, (SDL_Color){240, 180, 70, 255});
}

static void render_quantum(SDL_Renderer *r, const SDL_Rect *sim, const Params *p, const QuantumField *q) {
    (void)p;
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);

    float maxrho = 1e-8f;
    for (int j = 0; j < q->ny; ++j) {
        for (int i = 0; i < q->nx; ++i) {
            int id = IDX(i,j,q->nx);
            float rho = q->re[id]*q->re[id] + q->im[id]*q->im[id];
            if (rho > maxrho) maxrho = rho;
        }
    }

    float cw = (float)sim->w / (float)q->nx;
    float ch = (float)sim->h / (float)q->ny;

    for (int j = 0; j < q->ny; ++j) {
        for (int i = 0; i < q->nx; ++i) {
            int id = IDX(i,j,q->nx);
            float rho = q->re[id]*q->re[id] + q->im[id]*q->im[id];
            float t = sqrtf(rho / maxrho);
            Uint8 a = (Uint8)CLAMP((int)(255.0f * t), 0, 255);
            Uint8 blue = (Uint8)CLAMP((int)(210.0f * t + 30.0f), 0, 255);
            Uint8 green = (Uint8)CLAMP((int)(120.0f * t + 20.0f), 0, 255);

            SDL_Rect cell = {
                sim->x + (int)(i * cw),
                sim->y + (int)(j * ch),
                (int)(cw + 1.0f),
                (int)(ch + 1.0f)
            };
            SDL_SetRenderDrawColor(r, 50, green, blue, a);
            SDL_RenderFillRect(r, &cell);
        }
    }

    float xbar, ybar;
    quantum_expectation(q, &xbar, &ybar);
    int sx = sim->x + (int)((xbar / (q->dx * (q->nx - 1))) * sim->w);
    int sy = sim->y + (int)((ybar / (q->dy * (q->ny - 1))) * sim->h);
    draw_circle(r, sx, sy, 4, (SDL_Color){255, 255, 255, 255});

    draw_box(r, sim);
}

/* ------------------------------- Main app --------------------------------- */

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    g_window = SDL_CreateWindow(
        "Classical / Quantum Ball in a Gravitational Box",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN
    );
    if (!g_window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    g_renderer = SDL_CreateRenderer(g_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!g_renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(g_window);
        SDL_Quit();
        return 1;
    }

    Params p = {
        .angle_deg = 55.0f,
        .speed = 28.0f,
        .gravity = 18.0f,
        .box_w = 12.0f,
        .box_h = 8.0f,
    };

    ClassicalBall cball;
    QuantumField qfield;
    memset(&qfield, 0, sizeof(qfield));

    classical_reset(&cball, &p);
    quantum_reset(&qfield, &p);

    SimMode mode = MODE_CLASSICAL;
    bool paused = false;
    bool running = true;

    Slider sliders[5];
    int sx = 30, sy = 80, sw = CONTROL_W - 60, sh = 24, gap = 86;
    sliders[0] = (Slider){ .rect = {sx, sy + 0*gap, sw, sh}, .minv = 5.0f,  .maxv = 85.0f, .value = &p.angle_deg, .label = "ANGLE" };
    sliders[1] = (Slider){ .rect = {sx, sy + 1*gap, sw, sh}, .minv = 2.0f,  .maxv = 60.0f, .value = &p.speed,    .label = "SPEED" };
    sliders[2] = (Slider){ .rect = {sx, sy + 2*gap, sw, sh}, .minv = 0.0f,  .maxv = 45.0f, .value = &p.gravity,  .label = "GRAVITY" };
    sliders[3] = (Slider){ .rect = {sx, sy + 3*gap, sw, sh}, .minv = 6.0f,  .maxv = 20.0f, .value = &p.box_w,    .label = "BOX W" };
    sliders[4] = (Slider){ .rect = {sx, sy + 4*gap, sw, sh}, .minv = 4.0f,  .maxv = 14.0f, .value = &p.box_h,    .label = "BOX H" };

    Button btn_classical = {{30, 530, 120, 36}, "CLASSICAL", false};
    Button btn_quantum   = {{170, 530, 120, 36}, "QUANTUM", false};
    Button btn_reset     = {{30, 590, 120, 36}, "RESET", false};
    Button btn_pause     = {{170, 590, 120, 36}, "PAUSE", false};

    Uint64 prev = SDL_GetPerformanceCounter();
    double freq = (double)SDL_GetPerformanceFrequency();

    while (running) {
        bool need_reset = false;
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running = false;
            } else if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) {
                int mx = e.button.x, my = e.button.y;

                for (int i = 0; i < 5; ++i) {
                    if (point_in_rect(mx, my, sliders[i].rect)) {
                        sliders[i].dragging = true;
                        slider_handle_mouse(&sliders[i], mx);
                        need_reset = true;
                    }
                }

                if (point_in_rect(mx, my, btn_classical.rect)) {
                    mode = MODE_CLASSICAL;
                    need_reset = true;
                }
                if (point_in_rect(mx, my, btn_quantum.rect)) {
                    mode = MODE_QUANTUM;
                    need_reset = true;
                }
                if (point_in_rect(mx, my, btn_reset.rect)) {
                    need_reset = true;
                }
                if (point_in_rect(mx, my, btn_pause.rect)) {
                    paused = !paused;
                }
            } else if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT) {
                for (int i = 0; i < 5; ++i) sliders[i].dragging = false;
            } else if (e.type == SDL_MOUSEMOTION) {
                int mx = e.motion.x;
                for (int i = 0; i < 5; ++i) {
                    if (sliders[i].dragging) {
                        slider_handle_mouse(&sliders[i], mx);
                        need_reset = true;
                    }
                }
            } else if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE: running = false; break;
                    case SDLK_SPACE: paused = !paused; break;
                    case SDLK_r: need_reset = true; break;
                    case SDLK_c: mode = MODE_CLASSICAL; need_reset = true; break;
                    case SDLK_q: mode = MODE_QUANTUM; need_reset = true; break;
                    default: break;
                }
            }
        }

        if (need_reset) {
            classical_reset(&cball, &p);
            quantum_reset(&qfield, &p);
        }

        Uint64 now = SDL_GetPerformanceCounter();
        float dt = (float)((double)(now - prev) / freq);
        prev = now;
        dt = CLAMP(dt, 0.0f, 0.03f);

        if (!paused) {
            if (mode == MODE_CLASSICAL) {
                classical_step(&cball, &p, dt);
            } else {
                /* A few substeps per frame keep the PDE evolution smoother. */
                quantum_step(&qfield, 4);
            }
        }

        SDL_SetRenderDrawColor(g_renderer, 20, 20, 28, 255);
        SDL_RenderClear(g_renderer);

        SDL_Rect control = {0, 0, CONTROL_W, WINDOW_H};
        SDL_SetRenderDrawColor(g_renderer, 35, 35, 45, 255);
        SDL_RenderFillRect(g_renderer, &control);
        SDL_SetRenderDrawColor(g_renderer, 80, 80, 90, 255);
        SDL_RenderDrawLine(g_renderer, CONTROL_W, 0, CONTROL_W, WINDOW_H);

        draw_text(g_renderer, 28, 24, 3, "BALLISTIC BOX", (SDL_Color){245,245,245,255});
        draw_text(g_renderer, 28, 48, 2, "CLASSICAL OR QUANTUM", (SDL_Color){180,180,200,255});

        for (int i = 0; i < 5; ++i) draw_slider(g_renderer, &sliders[i]);
        draw_button(g_renderer, &btn_classical, mode == MODE_CLASSICAL);
        draw_button(g_renderer, &btn_quantum, mode == MODE_QUANTUM);
        draw_button(g_renderer, &btn_reset, false);
        draw_button(g_renderer, &btn_pause, paused);

        draw_text(g_renderer, 30, 660, 2, "KEYS: C Q R SPACE ESC", (SDL_Color){190,190,210,255});
        draw_text(g_renderer, 30, 700, 2, "WHITE DOT IN QUANTUM = <X>,<Y>", (SDL_Color){150,210,230,255});
        draw_text(g_renderer, 30, 730, 2, "HARD WALLS: PSI=0 ON BOUNDARY", (SDL_Color){150,210,230,255});
        draw_text(g_renderer, 30, 760, 2, "GRAVITY ACTS DOWNWARD", (SDL_Color){150,210,230,255});

        SDL_Rect sim = compute_sim_rect(&p);
        if (mode == MODE_CLASSICAL) {
            render_classical(g_renderer, &sim, &p, &cball);
        } else {
            render_quantum(g_renderer, &sim, &p, &qfield);
        }

        SDL_RenderPresent(g_renderer);
    }

    quantum_free(&qfield);
    SDL_DestroyRenderer(g_renderer);
    SDL_DestroyWindow(g_window);
    SDL_Quit();
    return 0;
}
