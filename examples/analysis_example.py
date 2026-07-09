from compas_acoustics import AcousticAnalyser
import soundfile as sf
import os

path = os.path.join(os.path.dirname(__file__), 'rir_mono.wav')
ir, fs = sf.read(path)

analyser = AcousticAnalyser(ir)
analyser.sampling_rate = fs
data = analyser.analyse()

freqs = analyser.frequency_bands
print(f"Frequency bands: {freqs}")

t30 = analyser.T30
print(f"T30: {t30}")

# --- print ---
analyser.print_table()
# analyser.print_table(show_overall=True)    
# analyser.print_table(parameters=['EDT', 'T20', 'T30'])
# analyser.print_table(parameters=['EDT', 'T20', 'T30'], show_overall=True)
# analyser.print_table(show_bands=False, show_overall=True)

# --- plot ---
analyser.plot_rir()
analyser.plot_frequency_response(smooth=3)
# analyser.plot() # plot all parameters
analyser.plot(parameters=['T20', 'C50', 'STI'])

# --- export ---
# csv_path = os.path.join(os.path.dirname(__file__), 'analysis_results.csv')
# analyser.export_csv(csv_path)
# analyser.export_csv(csv_path, parameters=['EDT', 'T20', 'T30'])
