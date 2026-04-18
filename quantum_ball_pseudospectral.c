#include <SDL2/SDL.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
    quantum_ball_pseudospectral.c

    Classical / quantum ballistic ball in a 2D rectangular box with gravity.

    Classical mode:
        - point-mass ballistic motion with elastic bounces at the walls

    Quantum mode:
        - 2D time-dependent Schrodinger equation in a hard-wall rectangle
        - split-step pseudospectral evolution
        - sine pseudospectral basis via FFTW DST-I transforms in x and y

    PDE:
        i hbar dpsi/dt = [ -(hbar^2/(2m)) Laplacian + m g y ] psi

    Because the walls are hard, psi = 0 on the boundary, so a sine basis is natural.
    One Strang step is:
        psi^{n+1} = exp(-i V dt / 2hbar) exp(-i T dt / hbar) exp(-i V dt / 2hbar) psi^n

    Build requires:
        SDL2
        FFTW3 single precision (fftw3f)

    Linux example:
        gcc -O2 -std=c11 quantum_ball_pseudospectral.c -o quantum_ball_pseudospectral \
            $(sdl2-config --cflags --libs) -lfftw3f -lm
*/

#define WINDOW_W 1400
#define WINDOW_H 900
#define CONTROL_W 320
#define MARGIN 16

#define CLAMP(x,a,b) ((x)<(a)?(a):((x)>(b)?(b):(x)))
#define IDX(i,j,nx) ((j)*(nx) + (i))
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef enum { MODE_CLASSICAL = 0, MODE_QUANTUM = 1 } SimMode;

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

typedef struct {
    int nx, ny;                 /* includes boundaries */
    int mx, my;                 /* interior sizes = nx-2, ny-2 */
    float Lx, Ly;
    float dx, dy;
    float dt;
    float mass;
    float hbar;

    float *re;                  /* full grid, size nx*ny */
    float *im;                  /* full grid, size nx*ny */

    float *spec_re;             /* interior spectral work arrays, size mx*my */
    float *spec_im;
    float *buf_x;
    float *buf_y;
    float *kin_phase_re;        /* exp(-i T dt / hbar) in spectral space */
    float *kin_phase_im;

    fftwf_plan dst_x_fwd;
    fftwf_plan dst_x_inv;
    fftwf_plan dst_y_fwd;
    fftwf_plan dst_y_inv;
} QuantumField;

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
    {'<',{0x02,0x04,0x08,0x10,0x08,0x04,0x02}},
    {'>',{0x08,0x04,0x02,0x01,0x02,0x04,0x08}},
    {'=',{0x00,0x1F,0x00,0x1F,0x00,0x00,0x00}},
    {'/',{0x01,0x02,0x04,0x08,0x10,0x00,0x00}},
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

static void format_float(char *buf, size_t n, const char *label, float v) {
    snprintf(buf, n, "%s %.2f", label, v);
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
    format_float(buf, sizeof(buf), s->label, *s->value);
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

/* ---------------------------- Quantum simulation --------------------------- */

static void quantum_free(QuantumField *q) {
    if (q->dst_x_fwd) fftwf_destroy_plan(q->dst_x_fwd);
    if (q->dst_x_inv) fftwf_destroy_plan(q->dst_x_inv);
    if (q->dst_y_fwd) fftwf_destroy_plan(q->dst_y_fwd);
    if (q->dst_y_inv) fftwf_destroy_plan(q->dst_y_inv);
    fftwf_free(q->re); fftwf_free(q->im);
    fftwf_free(q->spec_re); fftwf_free(q->spec_im);
    fftwf_free(q->buf_x); fftwf_free(q->buf_y);
    fftwf_free(q->kin_phase_re); fftwf_free(q->kin_phase_im);
    memset(q, 0, sizeof(*q));
}

static bool quantum_alloc(QuantumField *q, int nx, int ny) {
    memset(q, 0, sizeof(*q));
    q->nx = nx; q->ny = ny; q->mx = nx - 2; q->my = ny - 2;
    size_t nfull = (size_t)nx * (size_t)ny;
    size_t nint = (size_t)q->mx * (size_t)q->my;

    q->re = (float*)fftwf_alloc_real(nfull);
    q->im = (float*)fftwf_alloc_real(nfull);
    q->spec_re = (float*)fftwf_alloc_real(nint);
    q->spec_im = (float*)fftwf_alloc_real(nint);
    q->buf_x = (float*)fftwf_alloc_real((size_t)q->mx);
    q->buf_y = (float*)fftwf_alloc_real((size_t)q->my);
    q->kin_phase_re = (float*)fftwf_alloc_real(nint);
    q->kin_phase_im = (float*)fftwf_alloc_real(nint);
    if (!q->re || !q->im || !q->spec_re || !q->spec_im || !q->buf_x || !q->buf_y || !q->kin_phase_re || !q->kin_phase_im) {
        quantum_free(q);
        return false;
    }
    memset(q->re, 0, nfull*sizeof(float));
    memset(q->im, 0, nfull*sizeof(float));

    q->dst_x_fwd = fftwf_plan_r2r_1d(q->mx, q->buf_x, q->buf_x, FFTW_RODFT00, FFTW_MEASURE);
    q->dst_x_inv = fftwf_plan_r2r_1d(q->mx, q->buf_x, q->buf_x, FFTW_RODFT00, FFTW_MEASURE);
    q->dst_y_fwd = fftwf_plan_r2r_1d(q->my, q->buf_y, q->buf_y, FFTW_RODFT00, FFTW_MEASURE);
    q->dst_y_inv = fftwf_plan_r2r_1d(q->my, q->buf_y, q->buf_y, FFTW_RODFT00, FFTW_MEASURE);
    if (!q->dst_x_fwd || !q->dst_x_inv || !q->dst_y_fwd || !q->dst_y_inv) {
        quantum_free(q);
        return false;
    }
    return true;
}

static void quantum_build_kinetic_phase(QuantumField *q) {
    for (int j = 0; j < q->my; ++j) {
        int m = j + 1;
        float ky = (float)m * (float)M_PI / q->Ly;
        for (int i = 0; i < q->mx; ++i) {
            int n = i + 1;
            float kx = (float)n * (float)M_PI / q->Lx;
            float T = 0.5f * q->hbar * q->hbar / q->mass * (kx*kx + ky*ky);
            float phase = -T * q->dt / q->hbar;
            size_t id = (size_t)j * (size_t)q->mx + (size_t)i;
            q->kin_phase_re[id] = cosf(phase);
            q->kin_phase_im[id] = sinf(phase);
        }
    }
}

static void quantum_normalize(QuantumField *q) {
    double sum = 0.0;
    for (int j = 1; j < q->ny - 1; ++j) {
        for (int i = 1; i < q->nx - 1; ++i) {
            size_t id = IDX(i,j,q->nx);
            double rho = (double)q->re[id]*q->re[id] + (double)q->im[id]*q->im[id];
            sum += rho;
        }
    }
    sum *= (double)q->dx * (double)q->dy;
    if (sum <= 0.0) return;
    float s = (float)(1.0 / sqrt(sum));
    for (int j = 1; j < q->ny - 1; ++j) {
        for (int i = 1; i < q->nx - 1; ++i) {
            size_t id = IDX(i,j,q->nx);
            q->re[id] *= s;
            q->im[id] *= s;
        }
    }
}

static void quantum_init_packet(QuantumField *q, const Params *p) {
    float theta = p->angle_deg * (float)M_PI / 180.0f;
    float x0 = 0.15f * p->box_w;
    float y0 = 0.20f * p->box_h;
    float sigma = 0.06f * fminf(p->box_w, p->box_h);
    float px = q->mass * p->speed * cosf(theta);
    float py = -q->mass * p->speed * sinf(theta);

    memset(q->re, 0, (size_t)q->nx * (size_t)q->ny * sizeof(float));
    memset(q->im, 0, (size_t)q->nx * (size_t)q->ny * sizeof(float));

    for (int j = 1; j < q->ny - 1; ++j) {
        float y = j * q->dy;
        for (int i = 1; i < q->nx - 1; ++i) {
            float x = i * q->dx;
            float dx = x - x0;
            float dy = y - y0;
            float env = expf(-(dx*dx + dy*dy)/(2.0f*sigma*sigma));
            float ph = (px*x + py*y) / q->hbar;
            size_t id = IDX(i,j,q->nx);
            q->re[id] = env * cosf(ph);
            q->im[id] = env * sinf(ph);
        }
    }
    quantum_normalize(q);
}

static void quantum_reset(QuantumField *q, const Params *p) {
    quantum_free(q);

    int mx = (int)CLAMP((int)(p->box_w * 12.0f), 96, 240);
    int my = (int)CLAMP((int)(p->box_h * 12.0f), 96, 220);
    int nx = mx + 2;
    int ny = my + 2;
    if (!quantum_alloc(q, nx, ny)) {
        fprintf(stderr, "Quantum allocation/plan creation failed.\n");
        exit(1);
    }

    q->Lx = p->box_w;
    q->Ly = p->box_h;
    q->dx = q->Lx / (float)(q->nx - 1);
    q->dy = q->Ly / (float)(q->ny - 1);
    q->mass = 1.0f;
    q->hbar = 1.0f;
    q->dt = 0.0022f;

    quantum_build_kinetic_phase(q);
    quantum_init_packet(q, p);
}

static void quantum_half_potential_step(QuantumField *q, float gravity) {
    float coeff = +0.5f * q->dt * q->mass * gravity / q->hbar;
    for (int j = 1; j < q->ny - 1; ++j) {
        float y = j * q->dy;
        float ph = coeff * y;
        float c = cosf(ph);
        float s = sinf(ph);
        for (int i = 1; i < q->nx - 1; ++i) {
            size_t id = IDX(i,j,q->nx);
            float a = q->re[id], b = q->im[id];
            q->re[id] = c*a - s*b;
            q->im[id] = s*a + c*b;
        }
    }
}

static void dst2_interior(QuantumField *q, const float *src_full, float *dst_spec) {
    /* Copy interior */
    for (int j = 0; j < q->my; ++j) {
        for (int i = 0; i < q->mx; ++i) {
            dst_spec[(size_t)j * (size_t)q->mx + (size_t)i] = src_full[IDX(i+1,j+1,q->nx)];
        }
    }

    /* x transforms row-wise */
    for (int j = 0; j < q->my; ++j) {
        float *row = dst_spec + (size_t)j * (size_t)q->mx;
        memcpy(q->buf_x, row, (size_t)q->mx * sizeof(float));
        fftwf_execute(q->dst_x_fwd);
        memcpy(row, q->buf_x, (size_t)q->mx * sizeof(float));
    }

    /* y transforms column-wise */
    for (int i = 0; i < q->mx; ++i) {
        for (int j = 0; j < q->my; ++j) q->buf_y[j] = dst_spec[(size_t)j * (size_t)q->mx + (size_t)i];
        fftwf_execute(q->dst_y_fwd);
        for (int j = 0; j < q->my; ++j) dst_spec[(size_t)j * (size_t)q->mx + (size_t)i] = q->buf_y[j];
    }
}

static void idst2_to_full(QuantumField *q, const float *src_spec, float *dst_full) {
    size_t nint = (size_t)q->mx * (size_t)q->my;
    float *wrk = (float*)malloc(nint * sizeof(float));
    if (!wrk) {
        fprintf(stderr, "malloc failed in idst2_to_full\n");
        exit(1);
    }
    memcpy(wrk, src_spec, nint * sizeof(float));

    for (int i = 0; i < q->mx; ++i) {
        for (int j = 0; j < q->my; ++j) q->buf_y[j] = wrk[(size_t)j * (size_t)q->mx + (size_t)i];
        fftwf_execute(q->dst_y_inv);
        for (int j = 0; j < q->my; ++j) wrk[(size_t)j * (size_t)q->mx + (size_t)i] = q->buf_y[j];
    }
    for (int j = 0; j < q->my; ++j) {
        float *row = wrk + (size_t)j * (size_t)q->mx;
        memcpy(q->buf_x, row, (size_t)q->mx * sizeof(float));
        fftwf_execute(q->dst_x_inv);
        memcpy(row, q->buf_x, (size_t)q->mx * sizeof(float));
    }

    /* DST-I is self-inverse up to factor 2(N+1) per dimension */
    float scale = 1.0f / (4.0f * (float)(q->mx + 1) * (float)(q->my + 1));

    memset(dst_full, 0, (size_t)q->nx * (size_t)q->ny * sizeof(float));
    for (int j = 0; j < q->my; ++j) {
        for (int i = 0; i < q->mx; ++i) {
            dst_full[IDX(i+1,j+1,q->nx)] = wrk[(size_t)j * (size_t)q->mx + (size_t)i] * scale;
        }
    }
    free(wrk);
}

static void quantum_kinetic_step(QuantumField *q) {
    size_t nint = (size_t)q->mx * (size_t)q->my;
    float *new_re = (float*)malloc((size_t)q->nx * (size_t)q->ny * sizeof(float));
    float *new_im = (float*)malloc((size_t)q->nx * (size_t)q->ny * sizeof(float));
    if (!new_re || !new_im) {
        fprintf(stderr, "malloc failed in quantum_kinetic_step\n");
        exit(1);
    }

    dst2_interior(q, q->re, q->spec_re);
    dst2_interior(q, q->im, q->spec_im);

    for (size_t k = 0; k < nint; ++k) {
        float a = q->spec_re[k], b = q->spec_im[k];
        float c = q->kin_phase_re[k], s = q->kin_phase_im[k];
        q->spec_re[k] = c*a - s*b;
        q->spec_im[k] = s*a + c*b;
    }

    idst2_to_full(q, q->spec_re, new_re);
    idst2_to_full(q, q->spec_im, new_im);

    memcpy(q->re, new_re, (size_t)q->nx * (size_t)q->ny * sizeof(float));
    memcpy(q->im, new_im, (size_t)q->nx * (size_t)q->ny * sizeof(float));
    free(new_re);
    free(new_im);
}

static void quantum_step(QuantumField *q, float gravity, int substeps) {
    for (int s = 0; s < substeps; ++s) {
        quantum_half_potential_step(q, gravity);
        quantum_kinetic_step(q);
        quantum_half_potential_step(q, gravity);
    }
    quantum_normalize(q);
}

static void quantum_expectation(const QuantumField *q, float *xbar, float *ybar) {
    double sx = 0.0, sy = 0.0, norm = 0.0;
    for (int j = 1; j < q->ny - 1; ++j) {
        float y = j * q->dy;
        for (int i = 1; i < q->nx - 1; ++i) {
            float x = i * q->dx;
            size_t id = IDX(i,j,q->nx);
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
    int avail_w = WINDOW_W - CONTROL_W - 2*MARGIN;
    int avail_h = WINDOW_H - 2*MARGIN;
    float sx = (float)avail_w / p->box_w;
    float sy = (float)avail_h / p->box_h;
    float scale = fminf(sx, sy);
    SDL_Rect rc;
    rc.w = (int)(p->box_w * scale);
    rc.h = (int)(p->box_h * scale);
    rc.x = CONTROL_W + (avail_w - rc.w)/2 + MARGIN;
    rc.y = (avail_h - rc.h)/2 + MARGIN;
    return rc;
}

static void world_to_screen(const SDL_Rect *sim, const Params *p, float x, float y, int *sx, int *sy) {
    *sx = sim->x + (int)((x / p->box_w) * sim->w);
    *sy = sim->y + (int)((y / p->box_h) * sim->h);
}

static void draw_box(SDL_Renderer *r, const SDL_Rect *sim) {
    SDL_SetRenderDrawColor(r, 220,220,220,255);
    SDL_RenderDrawRect(r, sim);
}

static void render_classical(SDL_Renderer *r, const SDL_Rect *sim, const Params *p, const ClassicalBall *b) {
    draw_box(r, sim);
    int sx, sy;
    world_to_screen(sim, p, b->x, b->y, &sx, &sy);
    int sr = (int)(b->radius * (float)sim->w / p->box_w);
    if (sr < 3) sr = 3;
    draw_circle(r, sx, sy, sr, (SDL_Color){240,180,70,255});
}

static void render_quantum(SDL_Renderer *r, const SDL_Rect *sim, const QuantumField *q) {
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    float maxrho = 1e-10f;
    for (int j = 1; j < q->ny - 1; ++j) {
        for (int i = 1; i < q->nx - 1; ++i) {
            size_t id = IDX(i,j,q->nx);
            float rho = q->re[id]*q->re[id] + q->im[id]*q->im[id];
            if (rho > maxrho) maxrho = rho;
        }
    }

    float cw = (float)sim->w / (float)q->mx;
    float ch = (float)sim->h / (float)q->my;
    for (int j = 1; j < q->ny - 1; ++j) {
        for (int i = 1; i < q->nx - 1; ++i) {
            size_t id = IDX(i,j,q->nx);
            float rho = q->re[id]*q->re[id] + q->im[id]*q->im[id];
            float t = sqrtf(rho / maxrho);
            Uint8 a = (Uint8)CLAMP((int)(255.0f*t), 0, 255);
            Uint8 green = (Uint8)CLAMP((int)(100.0f + 120.0f*t), 0, 255);
            Uint8 blue  = (Uint8)CLAMP((int)(50.0f + 190.0f*t), 0, 255);
            SDL_Rect cell = {
                sim->x + (int)((i-1) * cw),
                sim->y + (int)((j-1) * ch),
                (int)(cw + 1.0f),
                (int)(ch + 1.0f)
            };
            SDL_SetRenderDrawColor(r, 30, green, blue, a);
            SDL_RenderFillRect(r, &cell);
        }
    }

    float xbar, ybar;
    quantum_expectation(q, &xbar, &ybar);
    int sx = sim->x + (int)((xbar / q->Lx) * sim->w);
    int sy = sim->y + (int)((ybar / q->Ly) * sim->h);
    draw_circle(r, sx, sy, 4, (SDL_Color){255,255,255,255});
    draw_box(r, sim);
}

/* --------------------------------- Main ----------------------------------- */

int main(void) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *window = SDL_CreateWindow(
        "Ballistic Box: Classical and Quantum (Split-Step Sine Pseudospectral)",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN
    );
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
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

    Slider sliders[5];
    int sx = 30, sy = 80, sw = CONTROL_W - 60, sh = 24, gap = 86;
    sliders[0] = (Slider){ .rect = {sx, sy + 0*gap, sw, sh}, .minv = 5.0f, .maxv = 85.0f, .value = &p.angle_deg, .label = "ANGLE" };
    sliders[1] = (Slider){ .rect = {sx, sy + 1*gap, sw, sh}, .minv = 2.0f, .maxv = 60.0f, .value = &p.speed, .label = "SPEED" };
    sliders[2] = (Slider){ .rect = {sx, sy + 2*gap, sw, sh}, .minv = 0.0f, .maxv = 45.0f, .value = &p.gravity, .label = "GRAVITY" };
    sliders[3] = (Slider){ .rect = {sx, sy + 3*gap, sw, sh}, .minv = 6.0f, .maxv = 20.0f, .value = &p.box_w, .label = "BOX W" };
    sliders[4] = (Slider){ .rect = {sx, sy + 4*gap, sw, sh}, .minv = 4.0f, .maxv = 14.0f, .value = &p.box_h, .label = "BOX H" };

    Button btn_classical = {{30, 530, 120, 36}, "CLASSICAL"};
    Button btn_quantum   = {{170, 530, 120, 36}, "QUANTUM"};
    Button btn_reset     = {{30, 590, 120, 36}, "RESET"};
    Button btn_pause     = {{170, 590, 120, 36}, "PAUSE"};

    SimMode mode = MODE_CLASSICAL;
    bool paused = false;
    bool running = true;

    Uint64 prev = SDL_GetPerformanceCounter();
    double freq = (double)SDL_GetPerformanceFrequency();

    while (running) {
        bool need_reset = false;
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
                if (point_in_rect(mx, my, btn_reset.rect))     { need_reset = true; }
                if (point_in_rect(mx, my, btn_pause.rect))     { paused = !paused; }
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
        float dt_wall = (float)((double)(now - prev) / freq);
        prev = now;
        dt_wall = CLAMP(dt_wall, 0.0f, 0.03f);

        if (!paused) {
            if (mode == MODE_CLASSICAL) {
                classical_step(&cball, &p, dt_wall);
            } else {
                quantum_step(&qfield, p.gravity, 3);
            }
        }

        SDL_SetRenderDrawColor(renderer, 20,20,28,255);
        SDL_RenderClear(renderer);
        SDL_Rect control = {0,0,CONTROL_W,WINDOW_H};
        SDL_SetRenderDrawColor(renderer, 35,35,45,255);
        SDL_RenderFillRect(renderer, &control);
        SDL_SetRenderDrawColor(renderer, 80,80,90,255);
        SDL_RenderDrawLine(renderer, CONTROL_W, 0, CONTROL_W, WINDOW_H);

        draw_text(renderer, 28, 24, 3, "BALLISTIC BOX", (SDL_Color){245,245,245,255});
        draw_text(renderer, 28, 48, 2, "CLASSICAL OR QUANTUM", (SDL_Color){180,180,200,255});
        for (int i = 0; i < 5; ++i) draw_slider(renderer, &sliders[i]);
        draw_button(renderer, &btn_classical, mode == MODE_CLASSICAL);
        draw_button(renderer, &btn_quantum, mode == MODE_QUANTUM);
        draw_button(renderer, &btn_reset, false);
        draw_button(renderer, &btn_pause, paused);
        draw_text(renderer, 30, 660, 2, "QUANTUM = STRANG SPLIT STEP", (SDL_Color){170,220,230,255});
        draw_text(renderer, 30, 690, 2, "SINE PSEUDOSPECTRAL / HARD WALLS", (SDL_Color){170,220,230,255});
        draw_text(renderer, 30, 720, 2, "KEYS C Q R SPACE ESC", (SDL_Color){190,190,210,255});
        draw_text(renderer, 30, 750, 2, "WHITE DOT = <X> <Y>", (SDL_Color){190,190,210,255});

        SDL_Rect sim = compute_sim_rect(&p);
        if (mode == MODE_CLASSICAL) render_classical(renderer, &sim, &p, &cball);
        else render_quantum(renderer, &sim, &qfield);

        SDL_RenderPresent(renderer);
    }

    quantum_free(&qfield);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
