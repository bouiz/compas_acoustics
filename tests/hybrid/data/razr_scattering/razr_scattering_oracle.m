% RAZR 0.96.3's surface scattering (Ewert et al. 2025) measured as a black box,
% for tests/hybrid/test_scattering.py. Read README.md first.
%
% Three parts, all at fs = 48 kHz, diotic, jitter, source and receiver smearing,
% air absorption and the FDN input allpass off, scat_cutoff 20 Hz (so
% |H_d|^2 = d over the audio band):
%
% cases     The four shoeboxes of razr_exact_rooms.m (2026-10-16), image source
%           order 3, the same d on every wall, d in {0.25, 0.5}:
%           specular_ratio(k), scattered_ratio(k)  per order k = 1..3, above 5 kHz
%           allpass_delays, mean_distance          the time-spread cascade
%           d_early, d_late, d_total, d_direct_early  level changes on vs off
%                                                  (informational, not tested)
% per_wall  Room 1 with d = [0.1 0.2 0.3 0.4 0.5 0.6] on the walls
%           [-z -y -x +x +y +z]: per image, its position and both ratios.
% coupling  The same room: the FDN input matrix solved for the gains of the
%           last-order specular signals and the smeared scattered signals,
%           with fdn_smart_ch_mapping 0 (RAZR's default) and 1.
%
% Run from MATLAB:
%   razr_scattering_oracle('razr_scattering_oracle.json', '/path/to/RAZR/build_0963')
function razr_scattering_oracle(outfile, razr_path)
addpath(razr_path);
razr_addpath;
cases = {
    [10 8 4],   0.15, [2.5 3.0 1.5], [7.5 5.5 1.2];
    [6 5 3],    0.10, [1.5 1.5 1.5], [4.5 3.5 1.2];
    [6 5 3],    0.20, [4.0 3.5 1.7], [2.0 2.0 1.5];
    [8 6 3.5],  0.10, [2.2 1.7 1.4], [5.9 4.1 1.3];
};
amounts = [0.25 0.5];
per_wall_d = [0.1 0.2 0.3 0.4 0.5 0.6];
ism_order = 3;
fs = 48000;
E = @(x) sum(x(:) .^ 2);
Ecol = @(x) sum(x .^ 2, 1);
[b, a] = butter(4, 5000 / (fs / 2), 'high');
hp = @(x) filter(b, a, x);

results = {};
for k = 1:size(cases, 1)
    for d = amounts
        room = make_room(cases(k, :), ones(1, 6) * d);
        op_on = make_op(fs, ism_order, 1);
        op_off = make_op(fs, ism_order, 0);
        [ir_on, ~, ~, data_on] = razr(room, op_on);
        [ir_off, ~, ~, data_off] = razr(room, op_off);
        check_images(ir_on, data_on, data_off);
        specular = zeros(1, ism_order);
        scattered = zeros(1, ism_order);
        for n = 1:ism_order
            c = data_on.order == n;
            reference = E(hp(ir_off.early_refl_sigmat(:, c)));
            specular(n) = E(hp(ir_on.early_refl_sigmat(:, c))) / reference;
            scattered(n) = E(hp(ir_on.early_refl_sigmat_diffuse(:, c))) / reference;
        end
        [~, ism_setup] = pipeline(room, op_on);
        r = struct();
        r.room = k;
        r.boxsize = cases{k, 1};
        r.absorption = cases{k, 2};
        r.source = cases{k, 3};
        r.receiver = cases{k, 4};
        r.d = d;
        r.specular_ratio = specular;
        r.scattered_ratio = scattered;
        r.mean_distance = mean([r.receiver, r.boxsize - r.receiver]);
        r.allpass_delays = cascade_delays(ism_setup.a_tspr);
        r.d_early = 10 * log10(E(ir_on.sig_early) / E(ir_off.sig_early));
        r.d_late = 10 * log10(E(ir_on.sig_late) / E(ir_off.sig_late));
        r.d_total = 10 * log10(E(ir_on.sig) / E(ir_off.sig));
        r.d_direct_early = 10 * log10(E(ir_on.sig_direct + ir_on.sig_early) / ...
            E(ir_off.sig_direct + ir_off.sig_early));
        results{end + 1} = r; %#ok<AGROW>
        fprintf('room %d d %.2f: specular %s scattered %s delays %s\n', ...
            k, d, mat2str(specular, 4), mat2str(scattered, 4), mat2str(r.allpass_delays));
    end
end

% Per wall: a different d on each wall, every image's own ratios
room = make_room(cases(1, :), per_wall_d);
op_on = make_op(fs, ism_order, 1);
op_off = make_op(fs, ism_order, 0);
[ir_on, ~, ~, data_on] = razr(room, op_on);
[ir_off, ~, ~, data_off] = razr(room, op_off);
check_images(ir_on, data_on, data_off);
reference = Ecol(hp(ir_off.early_refl_sigmat));
per_wall = struct();
per_wall.room = 1;
per_wall.walls = {'-z', '-y', '-x', '+x', '+y', '+z'};
per_wall.d = per_wall_d;
per_wall.order = data_on.order(:)';
per_wall.pattern = data_on.pattern;
per_wall.positions = data_on.positions;
per_wall.specular_ratio = Ecol(hp(ir_on.early_refl_sigmat)) ./ reference;
per_wall.scattered_ratio = Ecol(hp(ir_on.early_refl_sigmat_diffuse)) ./ reference;
per_wall.specular_ratio(per_wall.order == 0) = 1;
per_wall.scattered_ratio(per_wall.order == 0) = 0;

% Coupling: FDNinputmat = [last-order specular, smeared scattered] * G per channel
coupling = {};
for smart = [0 1]
    op = make_op(fs, ism_order, 1);
    op.fdn_smart_ch_mapping = smart;
    [X, ism_setup, ir, isd, fdn_setup] = pipeline(room, op);
    S = ir.curr_sigmat;
    reflected = isd.order(:)' > 0;  % the direct sound has no scattered part
    D = ir.early_refl_sigmat_diffuse(:, reflected);
    Dt = filter(ism_setup.b_tspr, ism_setup.a_tspr, D);
    P = fdn_setup.num_pre_zeros;
    delays = fdn_setup.delays(:)';
    numCh = size(X, 2);
    Gs = zeros(size(S, 2), numCh);
    Gd = zeros(numel(reflected), numCh);
    residual = 0;
    residual_unsmeared = 0;
    for ch = 1:numCh
        % the scattered parts enter one sample after skipping the line's delay
        A = [place(S, P, size(X, 1)), place(Dt, P - delays(ch) + 1, size(X, 1))];
        g = A \ X(:, ch);
        residual = residual + E(X(:, ch) - A * g);
        Gs(:, ch) = g(1:size(S, 2));
        Gd(reflected, ch) = g(size(S, 2) + 1:end);
        Au = [place(S, P, size(X, 1)), place(D, P - delays(ch) + 1, size(X, 1))];
        residual_unsmeared = residual_unsmeared + E(X(:, ch) - Au * (Au \ X(:, ch)));
    end
    c = struct();
    c.fdn_smart_ch_mapping = smart;
    c.residual = residual / E(X);
    c.residual_unsmeared = residual_unsmeared / E(X);
    c.num_pre_zeros = P;
    c.delays = delays;
    c.azim = fdn_setup.azim(:)';
    c.elev = fdn_setup.elev(:)';
    c.order = isd.order(:)';
    c.relpos = isd.relpos;
    c.is_fdn_input = isd.is_fdn_input(:)';
    c.specular_gains = Gs;
    c.scattered_gains = Gd;
    coupling{end + 1} = c; %#ok<AGROW>
    fprintf('coupling, smart mapping %d: residual %.3g (unsmeared %.3g)\n', smart, c.residual, c.residual_unsmeared);
end

out = struct();
out.razr_version = get_razr_version;
out.razr_build = 'build_0963 (0.96.3)';
out.matlab_version = version;
out.options = struct('fs', fs, 'ism_order', ism_order, 'spat_mode', 'diotic', ...
    'ism_jitter_factor', 0, 'enableSR', [0 0], 'ism_enableAirAbsFilt', 0, ...
    'fdn_enable_apc', 0, 'ism_enable_scattering', [1 0], 'scat_cutoff_hz', 20, 'band_above_hz', 5000);
out.cases = results;
out.per_wall = per_wall;
out.coupling = coupling;
fid = fopen(outfile, 'w');
fprintf(fid, '%s\n', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
end

function room = make_room(c, scat)
room = struct();
room.boxsize = c{1};
room.freq = [125 250 500 1000 2000 4000];
room.materials = c{2};
room.srcpos = c{3};
room.recpos = c{4};
room.recdir = [0 0];
room.scat_amount = scat;
room.scat_cutoff = ones(1, 6) * 20;
end

function op = make_op(fs, ism_order, scattering)
op = struct();
op.fs = fs;
op.ism_order = ism_order;
op.spat_mode = 'diotic';
op.ism_jitter_factor = 0;
op.enableSR = [0 0];
op.ism_enableAirAbsFilt = 0;
op.fdn_enable_apc = 0;
op.ism_enable_scattering = scattering;
op.return_rir_parts = 1;
op.return_ism_data = 1;
op.return_ism_sigmat = 1;
op.verbosity = 0;
end

function check_images(ir_on, data_on, data_off)
if ~isequal(data_on.order, data_off.order)
    error('the image sources differ with scattering on and off');
end
if size(ir_on.early_refl_sigmat, 2) ~= numel(data_on.order) || ...
        size(ir_on.early_refl_sigmat_diffuse, 2) ~= numel(data_on.order)
    error('the signal matrices do not hold one column per image source');
end
end

function [X, ism_setup, ir, isd, fdn_setup] = pipeline(room, op)
% create_rir.m's steps up to the FDN input, which razr() does not return
% (op.debugMode would store it, but fails in getReflFiltCoeff.m:56).
op = complement_options(op);
op.array_pos_rec = op.array_pos;
op = rotate_array(room, op);
room = complement_room(room, op);
ism_setup = get_ism_setup(room, op);
[ir, isd] = image_source_model(room, ism_setup, op);
fdn_setup = get_fdn_setup(room, op);
[X, fdn_setup] = gen_fdn_input(ir, ism_setup, fdn_setup, isd, room, op);
end

function out = place(A, lag, len)
out = zeros(len, size(A, 2));
lo = max(lag, 0) + 1;
hi = min(lag + size(A, 1), len);
out(lo:hi, :) = A(lo - lag:hi - lag, :);
end

function delays = cascade_delays(a)
% The denominator of a cascade of (g + z^-tau) / (1 + g z^-tau) is
% prod(1 + g z^-tau_i); its nonzero taps sit at the subset sums of the delays.
taps = find(abs(a(2:end)) > 1e-12);
delays = [];
sums = 0;
for t = taps(:)'
    if ~ismember(t, sums)
        delays(end + 1) = t; %#ok<AGROW>
        sums = unique([sums, sums + t]);
    end
end
delays = sort(delays, 'descend');
end
