from compas_acoustics import AcousticAnalyser
import soundfile as sf
import os

path = os.path.join(os.path.dirname(__file__), 'rir_mono.wav')
ir, fs = sf.read(path)

analyzer = AcousticAnalyser(ir)
analyzer.sampling_rate = fs
results = analyzer.analyse()

# analyzer.print_table()

analyzer.plot_rir()
analyzer.plot_frequency_response(smooth=3)

analyzer.plot('T30')


# csv_path = os.path.join(os.path.dirname(__file__), "analysis_results.csv")
# analyzer.export_csv(csv_path)