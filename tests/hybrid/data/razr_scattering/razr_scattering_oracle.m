% RAZR 0.96.3's surface scattering (Ewert et al. 2025) measured as a black box,
% for tests/hybrid/test_scattering.py.
%
% The four shoeboxes of the hybrid transition validation (razr_exact_rooms.m,
% 2026-10-16), image source order 3, scattering amount d in {0.25, 0.5} on every
% wall with a cutoff of 20 Hz, so |H_d|^2 = d over the audio band. Jitter, source
% and receiver smearing, air absorption and the FDN input allpass are off.
%
% Stored per room and d:
%   specular_ratio(k)   energy of the order-k specular images, on / off
%   scattered_ratio(k)  energy of the order-k scattered parts / order-k specular off
%                       (both above 5 kHz, k = 1..3)
%   d_early, d_late, d_total, d_direct_early
%                       energy change scattering on vs off of sig_early,
%                       sig_late, sig and sig_direct + sig_early, in dB
%
% Run from MATLAB: razr_scattering_oracle('razr_scattering_oracle.json')
function razr_scattering_oracle(outfile)
addpath('/Users/gpapadim/Documents/TOOLBOX/RAZR/build_0963');
razr_addpath;
cases = {
    [10 8 4],   0.15, [2.5 3.0 1.5], [7.5 5.5 1.2];
    [6 5 3],    0.10, [1.5 1.5 1.5], [4.5 3.5 1.2];
    [6 5 3],    0.20, [4.0 3.5 1.7], [2.0 2.0 1.5];
    [8 6 3.5],  0.10, [2.2 1.7 1.4], [5.9 4.1 1.3];
};
amounts = [0.25 0.5];
ism_order = 3;
fs = 48000;
E = @(x) sum(x(:) .^ 2);
[b, a] = butter(4, 5000 / (fs / 2), 'high');
hp = @(x) filter(b, a, x);

results = {};
for k = 1:size(cases, 1)
    for d = amounts
        room = struct();
        room.boxsize = cases{k, 1};
        room.freq = [125 250 500 1000 2000 4000];
        room.materials = cases{k, 2};
        room.srcpos = cases{k, 3};
        room.recpos = cases{k, 4};
        room.recdir = [0 0];
        room.scat_amount = ones(1, 6) * d;
        room.scat_cutoff = ones(1, 6) * 20;
        op = struct();
        op.fs = fs;
        op.ism_order = ism_order;
        op.spat_mode = 'diotic';
        op.ism_jitter_factor = 0;
        op.enableSR = [0 0];
        op.ism_enableAirAbsFilt = 0;
        op.fdn_enable_apc = 0;
        op.return_rir_parts = 1;
        op.return_ism_data = 1;
        op.return_ism_sigmat = 1;
        op.verbosity = 0;
        op_on = op; op_on.ism_enable_scattering = 1;
        op_off = op; op_off.ism_enable_scattering = 0;
        [ir_on, ~, ~, data_on] = razr(room, op_on);
        [ir_off, ~, ~, data_off] = razr(room, op_off);
        if ~isequal(data_on.order, data_off.order)
            error('the image sources differ with scattering on and off');
        end
        if size(ir_on.early_refl_sigmat, 2) ~= numel(data_on.order) || ...
                size(ir_on.early_refl_sigmat_diffuse, 2) ~= numel(data_on.order)
            error('the signal matrices do not hold one column per image source');
        end
        specular = zeros(1, ism_order);
        scattered = zeros(1, ism_order);
        for n = 1:ism_order
            c = data_on.order == n;
            reference = E(hp(ir_off.early_refl_sigmat(:, c)));
            specular(n) = E(hp(ir_on.early_refl_sigmat(:, c))) / reference;
            scattered(n) = E(hp(ir_on.early_refl_sigmat_diffuse(:, c))) / reference;
        end
        r = struct();
        r.room = k;
        r.boxsize = cases{k, 1};
        r.absorption = cases{k, 2};
        r.source = cases{k, 3};
        r.receiver = cases{k, 4};
        r.d = d;
        r.specular_ratio = specular;
        r.scattered_ratio = scattered;
        r.d_early = 10 * log10(E(ir_on.sig_early) / E(ir_off.sig_early));
        r.d_late = 10 * log10(E(ir_on.sig_late) / E(ir_off.sig_late));
        r.d_total = 10 * log10(E(ir_on.sig) / E(ir_off.sig));
        r.d_direct_early = 10 * log10(E(ir_on.sig_direct + ir_on.sig_early) / ...
            E(ir_off.sig_direct + ir_off.sig_early));
        results{end + 1} = r; %#ok<AGROW>
        fprintf('room %d d %.2f: specular %s scattered %s early %.3f late %.3f total %.3f dB\n', ...
            k, d, mat2str(specular, 4), mat2str(scattered, 4), r.d_early, r.d_late, r.d_total);
    end
end

out = struct();
out.razr_version = get_razr_version;
out.matlab_version = version;
out.options = struct('fs', fs, 'ism_order', ism_order, 'spat_mode', 'diotic', ...
    'ism_jitter_factor', 0, 'enableSR', [0 0], 'ism_enableAirAbsFilt', 0, ...
    'fdn_enable_apc', 0, 'scat_cutoff_hz', 20, 'band_above_hz', 5000);
out.cases = results;
fid = fopen(outfile, 'w');
fprintf(fid, '%s\n', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
end
