% LF and LFC of the plan's synthetic cases by the ITA Toolbox (ita_roomacoustics_lateral, ISO 3382-1),
% written to ita_lf_cases.json. Omni and figure-of-eight: a direct sound at 2 m (null towards it) and one
% reflection of amplitude 0.5 at 20 ms, lateral (case 'lateral') or at 45 deg (case 'oblique'), or lateral
% reflections at 3 ms and 90 ms only (case 'outside'), fs 48 kHz, 1 s long.
function ita_lf_cases(outfile)
fs = 48000; n = fs; c = 343.98;
t0 = round(2 / c * fs) + 1;
cases = {'lateral', [0.020 0.5 1.0]; 'oblique', [0.020 0.5 sqrt(0.5)]; 'outside', [0.003 0.5 1.0; 0.090 0.5 1.0]};
out = struct();
for k = 1:size(cases, 1)
    omni = zeros(n, 1); eight = zeros(n, 1);
    omni(t0) = 1;
    refl = cases{k, 2};
    for r = 1:size(refl, 1)
        i = t0 + round(refl(r, 1) * fs);
        omni(i) = omni(i) + refl(r, 2);
        eight(i) = eight(i) + refl(r, 2) * refl(r, 3);
    end
    a = itaAudio([omni eight], fs, 'time');
    a.channelNames = {'Omni', 'Eight'};
    [lfc, lf] = ita_roomacoustics_lateral(a, 'freqRange', [125 8000], 'bandsPerOctave', 1);
    out.(cases{k, 1}) = struct('freqs', lf.freqVector(:)', 'LF', lf.freqData(:)', 'LFC', lfc.freqData(:)');
end
fid = fopen(outfile, 'w'); fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true)); fclose(fid);
end
