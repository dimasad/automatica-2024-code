%% Load datafile
d = load(matfile);
sys = ss(d.A, d.B, d.C, d.D, -1);

%% Unpack model dimensions
nx = length(d.A);
nu = size(d.u, 2);
ny = size(d.y, 2);
if not(exist('N', 'var'))
    N = length(d.y);
end
fprintf("N: %f\n", N)

%% Create identification data structure
z = iddata(d.y(1:N,:), d.u(1:N, :));
clear d

%% Estimate the model
opt = ssestOptions('Display', 'on');

tic
if exist("nullsys0", "var") && nullsys0
    sys0 = idss(zeros(nx), zeros(nx, nu), zeros(ny,nx), zeros(ny,nu), ...
                randn(nx,ny)*1e-3, 'Ts', -1);
    sys0.Structure.K.Free = true;
    sys0.Structure.D.Free = true;
    syse = ssest(z, sys0, opt);
else
    syse = ssest(z, nx, 'Ts', -1, 'Feedthrough', 1, opt);
end
time = toc;

eratio = impulse_err(sys, syse);
niter = syse.Report.Termination.Iterations;

%% Display and save results
fprintf("Elapsed: %f s\n", time)
fprintf("Eratio: %f\n", eratio)
fprintf("niter: %f\n", niter)

dataout = [N, time, eratio, niter];
save(outfile, "dataout", "-ascii")

%% Auxiliary functions
function eratio = impulse_err(sys_true, sys_est)
% Impulse response error ratio
h_true = impulse(sys_true, 100);
h_est = impulse(sys_est, 100);
err = h_true - h_est;

norm_err = sqrt(sum(err.^2, 1));
norm_sig = sqrt(sum(h_true.^2, 1));
eratio = mean(norm_err ./ norm_sig, "all");
end