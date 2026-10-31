% quantum_ball_modal_octave.m
%
% Octave-robust pure spectral / Galerkin simulation of a quantum ballistic
% packet in a 2D rectangular box with hard walls and gravity.
%
% PDE:
%   i hbar d_t psi = [-(hbar^2 / 2m) Laplacian - m g y] psi
%
% Domain:
%   0 < x < Lx, 0 < y < Ly
%   psi = 0 on all walls
%
% Method:
%   - Expand psi in the sine basis satisfying Dirichlet BCs exactly.
%   - Since V depends only on y, propagate each x-mode independently:
%         c_n(t+dt) = exp(-i Ex_n dt / hbar) * Uy * c_n(t)
%     where Uy = expm(-i Hy dt / hbar) is precomputed once.
%   - Reconstruct |psi|^2 on a plotting grid for animation.
%
% This version is written to behave well in GNU Octave:
%   - no parula dependency
%   - persistent plot handles
%   - explicit complex arrays
%   - conservative figure updating with drawnow
%
% Run:
%   quantum_ball_modal_octave

clear;
clc;
close all;

%% --------------------------- Saving the video  ---------------------------
save_frames = true;
frame_dir = "frames_quantum_ball";

if save_frames
  if exist(frame_dir, "dir") ~= 7
    mkdir(frame_dir);
  end
end

%% --------------------------- Graphics toolkit ----------------------------
% Try qt first if available, otherwise fall back quietly.
try
  tks = available_graphics_toolkits();
  if any(strcmp(tks, "qt"))
    graphics_toolkit("qt");
  elseif any(strcmp(tks, "fltk"))
    graphics_toolkit("fltk");
  elseif any(strcmp(tks, "gnuplot"))
    graphics_toolkit("gnuplot");
  end
catch
  % Do nothing if toolkit probing is unavailable.
end

%% ----------------------------- Parameters ---------------------------------
% Screen-style coordinates: y increases downward.
% Therefore V(y) = -m g y produces downward acceleration.

hbar = 1.0;
mass = 1.0;
g    = 1.0;

Lx = 36.0;
Ly = 18.0;

x0 = 0.15 * Lx;
y0 = 0.20 * Ly;
angle_deg = 45.0;
v0 = 1.0;
sigma = 1.0;

theta = angle_deg * pi / 180.0;
px0 = mass * v0 * cos(theta);
py0 = -mass * v0 * sin(theta);

Nx = 30;
Ny = 30;

fps = 30;
dt = 1/60;
Tfinal = 30.0;
steps = round(Tfinal / dt);
frame_every = max(1, round((1/fps) / dt));

Nx_plot = 480;
Ny_plot = 480;
xg = linspace(0, Lx, Nx_plot);
yg = linspace(0, Ly, Ny_plot);
[Xg, Yg] = meshgrid(xg, yg);

%% ---------------------- Build sine basis on plot grid ---------------------
Sx_plot = zeros(Nx_plot, Nx);
Sy_plot = zeros(Ny_plot, Ny);
for n = 1:Nx
  Sx_plot(:, n) = sin(n * pi * xg(:) / Lx);
end
for m = 1:Ny
  Sy_plot(:, m) = sin(m * pi * yg(:) / Ly);
end

%% ---------------------- Build the modal Hamiltonian -----------------------
nvec = (1:Nx).';
Ex = (hbar^2 / (2 * mass)) * (nvec * pi / Lx).^2;

mvec = (1:Ny).';
Ey = (hbar^2 / (2 * mass)) * (mvec * pi / Ly).^2;

Ymat = zeros(Ny, Ny);
for mp = 1:Ny
  for m = 1:Ny
    if mp == m
      Ymat(mp, m) = Ly / 2;
    else
      if mod(mp + m, 2) == 0
        Ymat(mp, m) = 0.0;
      else
        Ymat(mp, m) = (2 * Ly / pi^2) * (1 / (mp + m)^2 - 1 / (mp - m)^2);
      end
    end
  end
end

Hy = diag(Ey) - mass * g * Ymat;
Uy = complex(expm((-1i / hbar) * Hy * dt));

%% --------------------- Initial modal coefficients c(n,m) ------------------
% Compute coefficients by numerical quadrature against the sine basis.
Nqx = 280;
Nqy = 220;
xq = linspace(0, Lx, Nqx);
yq = linspace(0, Ly, Nqy);
[Xq, Yq] = meshgrid(xq, yq);

psi0 = exp(-((Xq - x0).^2 + (Yq - y0).^2) / (4 * sigma^2)) .* ...
       exp(1i * (px0 * Xq + py0 * Yq) / hbar);

psi0(:, 1)   = 0;
psi0(:, end) = 0;
psi0(1, :)   = 0;
psi0(end, :) = 0;

Sx_q = zeros(Nqx, Nx);
Sy_q = zeros(Nqy, Ny);
for n = 1:Nx
  Sx_q(:, n) = sin(n * pi * xq(:) / Lx);
end
for m = 1:Ny
  Sy_q(:, m) = sin(m * pi * yq(:) / Ly);
end

C = complex(zeros(Nx, Ny));
for n = 1:Nx
  for m = 1:Ny
    basis_nm = Sy_q(:, m) * (Sx_q(:, n).');
    integrand = psi0 .* basis_nm;
    val = trapz(yq, trapz(xq, integrand, 2));
    C(n, m) = (4 / (Lx * Ly)) * val;
  end
end

modal_norm = (Lx * Ly / 4) * sum(abs(C(:)).^2);
C = C / sqrt(modal_norm);

%% ---------------------- Classical reference with bounces -------------------
traj_t = (0:steps) * dt;
x_class = zeros(1, steps + 1);
y_class = zeros(1, steps + 1);

x_class(1) = x0;
y_class(1) = y0;
vx_class = v0 * cos(theta);
vy_class = -v0 * sin(theta);

ball_r = 0.0;   % point reference; set >0 if you want finite-radius wall offset

for k = 1:steps
  vy_class = vy_class + g * dt;
  x_new = x_class(k) + vx_class * dt;
  y_new = y_class(k) + vy_class * dt;

  if x_new < ball_r
    x_new = ball_r + (ball_r - x_new);
    vx_class = -vx_class;
  elseif x_new > Lx - ball_r
    x_new = (Lx - ball_r) - (x_new - (Lx - ball_r));
    vx_class = -vx_class;
  end

  if y_new < ball_r
    y_new = ball_r + (ball_r - y_new);
    vy_class = -vy_class;
  elseif y_new > Ly - ball_r
    y_new = (Ly - ball_r) - (y_new - (Ly - ball_r));
    vy_class = -vy_class;
  end

  x_class(k + 1) = x_new;
  y_class(k + 1) = y_new;
end

%% --------------------------- Helper: reconstruction ------------------------
function [Psi, Rho, plot_norm, xbar, ybar] = reconstruct_state(C, Sy_plot, Sx_plot, Xg, Yg, xg, yg)
  Psi = Sy_plot * (C.') * Sx_plot.';
  Rho = abs(Psi).^2;
  plot_norm = trapz(yg, trapz(xg, Rho, 2));
  xbar = trapz(yg, trapz(xg, Rho .* Xg, 2)) / plot_norm;
  ybar = trapz(yg, trapz(xg, Rho .* Yg, 2)) / plot_norm;
end

%% ------------------------------ Initial plot -------------------------------
[Psi, Rho, plot_norm, xbar, ybar] = reconstruct_state(C, Sy_plot, Sx_plot, Xg, Yg, xg, yg);

fig = figure("Color", "w", "Name", "Quantum ballistic packet in a box");

if exist("parula", "file") || exist("parula", "builtin")
  colormap(parula(256));
else
  colormap(jet(256));
end

ax = axes("Parent", fig);
hImg = imagesc(xg, yg, Rho, "Parent", ax);
set(ax, "YDir", "reverse");
axis(ax, "image");
axis(ax, [0 Lx 0 Ly]);

% rho_max_fixed = max(Rho(:));
% if rho_max_fixed <= 0
%   rho_max_fixed = 1;
% end
% caxis(ax, [0 rho_max_fixed]);

hold(ax, "on");
rectangle("Position", [0 0 Lx Ly], "EdgeColor", "k", "LineWidth", 2);

hCentroid = plot(ax, xbar, ybar, "wo", "MarkerFaceColor", "w", "MarkerSize", 7);
hClassTrail = plot(ax, x_class(1), y_class(1), "r--", "LineWidth", 1.2);
hClassPoint = plot(ax, x_class(1), y_class(1), "ro", "MarkerFaceColor", "r", "MarkerSize", 6);

xlabel(ax, "x");
ylabel(ax, "y (downward)");
colorbar(ax);
title(ax, {
  sprintf("Pure spectral evolution in sine basis, t = %.3f", 0.0), ...
  sprintf("Nx=%d, Ny=%d, hbar=%.3f, m=%.3f, g=%.3f, ||psi||^2≈%.6f", Nx, Ny, hbar, mass, g, plot_norm), ...
  sprintf("<x>=%.3f, <y>=%.3f, angle=%.1f deg, v0=%.2f", xbar, ybar, angle_deg, v0)
});
drawnow();

%% ------------------------------- Animation ---------------------------------
for step = 0:steps
  t = step * dt;

  if mod(step, frame_every) == 0 || step == steps
    [Psi, Rho, plot_norm, xbar, ybar] = reconstruct_state(C, Sy_plot, Sx_plot, Xg, Yg, xg, yg);

    set(hImg, "CData", Rho);
    set(hCentroid, "XData", xbar, "YData", ybar);

    xc = x_class(1:(step + 1));
    yc = y_class(1:(step + 1));
    set(hClassTrail, "XData", xc, "YData", yc);
    set(hClassPoint, "XData", x_class(step + 1), "YData", y_class(step + 1));

    title(ax, {
      sprintf("Pure spectral evolution in sine basis, t = %.3f", t), ...
      sprintf("Nx=%d, Ny=%d, hbar=%.3f, m=%.3f, g=%.3f, ||psi||^2≈%.6f", Nx, Ny, hbar, mass, g, plot_norm), ...
      sprintf("<x>=%.3f, <y>=%.3f, classical=(%.3f, %.3f)", xbar, ybar, x_class(step + 1), y_class(step + 1))
    });

    ## if mod(step, 20) == 0
    ##   fprintf("step=%d  t=%.3f  xbar=%.5f  ybar=%.5f  norm=%.8f\n", step, t, xbar, ybar, plot_norm);
    ## end

    drawnow();
    if save_frames
      fr = getframe(fig);
      img = fr.cdata;
      fname = sprintf("%s/frame_%05d.png", frame_dir, floor(step / frame_every));
      imwrite(img, fname);
    end
  end

  if step < steps
    Cold = C;
    for n = 1:Nx
      C(n, :) = (exp(-1i * Ex(n) * dt / hbar) * (Uy * C(n, :).')).';
    end

    ## if mod(step, 20) == 0
    ##   fprintf("  update_size=%.6e\n", norm(C(:) - Cold(:)));
    ## end
  end
end

%% ------------------------------- Diagnostics -------------------------------
[Psi, Rho, plot_norm, xbar, ybar] = reconstruct_state(C, Sy_plot, Sx_plot, Xg, Yg, xg, yg);

%fprintf("Final plot-grid norm = %.10f\n", plot_norm);
%fprintf("Final expectation x  = %.10f\n", xbar);
%fprintf("Final expectation y  = %.10f\n", ybar);
