from compas_acoustics import AcousticAnalyser
import soundfile as sf
import os

path = os.path.join(os.path.dirname(__file__), 'rir_mono.wav')
ir, fs = sf.read(path)

analyzer = AcousticAnalyser(ir)
analyzer.sampling_rate = fs
analyzer.frequency_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
results = analyzer.analyse()

analyzer.print_table()

analyzer.plot('T30')

csv_path = os.path.join(os.path.dirname(__file__), "analysis_results.csv")
analyzer.export_csv(csv_path)