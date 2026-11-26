# Anechoic Soundfiles Library

Anechoic source recordings used by `compas_acoustics` for auralisation and related demos. All are mono.

## Files in this folder

| File | Description | Format | Source |
| --- | --- | --- | --- |
| `viola.wav` | Modern viola, C major detache | 48 kHz, 16 bit, 11.4 s | [OpenAIR, modern-viola-c-major-detache.wav](https://webfiles.york.ac.uk/OPENAIR/Anechoic/modern-viola/mono/) |
| `female_speech.wav` | Female speech | 48 kHz, float, 13.3 s | [EARS dataset](https://sp-uhh.github.io/ears_dataset/) |
| `male_speech.wav` | Male speech | 48 kHz, float, 12.5 s | [EARS dataset](https://sp-uhh.github.io/ears_dataset/) |
| `guitar.wav` | Flamenco guitar, anechoic (Flamenco piece 2, Neumann U89) | 48 kHz, 24 bit, 14.4 s | [TH Koeln Audio Group, Anechoic Recordings](https://www.audiogroup.web.th-koeln.de/anechoic.html), Flamenco2_U89.wav |
| `castanets.wav` | Castanets rhythm, anechoic | 48 kHz, float, 2.2 s | [Matthias Frank, IEM Graz](https://iaem.at/Members/frank/sounds/) |

`viola.wav` is the 96 kHz original resampled to 48 kHz (polyphase, anti-aliased), the sampling rate of the package.
`guitar.wav` is the excerpt from 5.13 s to 19.51 s of the original (the start is where pyfar's `guitar()` excerpt starts, the end the
pause at 19.5 s), with a 5 ms fade in and a 50 ms fade out, at its original level. `castanets.wav` is the file of pyfar
(`pyfar.signals.files.castanets`) at 44.1 kHz resampled to 48 kHz (polyphase, anti-aliased); it is stored as float because the
resampled peak is +0.06 dBFS.

## OpenAIR files: licence and attribution

`viola.wav` comes from the [OpenAIR Anechoic Data](https://www.openair.hosted.york.ac.uk/?page_id=310) collection (University of York),
which publishes recordings captured under anechoic conditions: instruments, singing, speech and other sources. It is licensed under a
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

If you use these recordings, please attribute them to [www.openairlib.net](https://www.openairlib.net) / [OpenAIR Anechoic Data](https://www.openair.hosted.york.ac.uk/?page_id=310).

## EARS files: licence and attribution

`female_speech.wav` and `male_speech.wav` come from the [EARS dataset](https://sp-uhh.github.io/ears_dataset/), an anechoic fullband speech dataset.
The README of its [repository](https://github.com/facebookresearch/ears_dataset) states that the dataset is released under the "CC-NC 4.0 International license": Creative Commons NonCommercial, which does not allow
commercial use. This is stricter than the licence of the rest of this package, so check it before the package is used or distributed commercially.

Cite: Richter, J., Wu, Y.-C., Krenn, S., Welker, S., Lay, B., Watanabe, S., Richard, A., Gerkmann, T. (2024). "EARS: An Anechoic Fullband Speech Dataset
Benchmarked for Speech Enhancement and Dereverberation." Interspeech 2024.

## TH Koeln file: licence and attribution

`guitar.wav` comes from the [Anechoic Recordings](https://www.audiogroup.web.th-koeln.de/anechoic.html) of the Audio Group, Cologne
University of Applied Sciences (TH Koeln), by Michio Woirgard, Philipp Stade, Jeffrey Amankwor, Benjamin Bernschuetz and Johannes Arend
(file Flamenco2_U89.wav). It is licensed under a [Creative Commons Attribution-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-sa/3.0/)
(CC BY-SA 3.0). ShareAlike applies to this file and its derivatives: they must be distributed under the same licence.

## IEM file: licence and attribution

`castanets.wav` is by Matthias Frank, IEM Graz ([sounds](https://iaem.at/Members/frank/sounds/)): the castanets rhythm of track 27 of the
EBU SQAM CD, re-programmed as an anechoic version with samples of the Vienna Symphonic Library. It is released under
[CC0](https://creativecommons.org/publicdomain/zero/1.0/) (Matthias Frank).
