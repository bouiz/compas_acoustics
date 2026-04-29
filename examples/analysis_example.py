from compas_acoustics import AcousticAnalyser
import soundfile as sf

ir, fs = sf.read("/Users/gpapadim/Desktop/odeon_srir.wav")
ir = ir[:, 0]

analyzer = AcousticAnalyser(ir)
analyzer.sampling_rate = fs
analyzer.frequency_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
results = analyzer.analyse()

analyzer.print_table()

analyzer.plotter('T30')

analyzer.export_csv("analysis_results.csv")