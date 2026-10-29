# Anechoic Soundfiles Library

Anechoic source recordings used by `compas_acoustics` for auralisation and related demos. All are mono.

## Files in this folder

| File | Description | Format | Source |
| --- | --- | --- | --- |
| `viola.wav` | Modern viola, C major detache | 48 kHz, 16 bit, 11.4 s | [OpenAIR, modern-viola-c-major-detache.wav](https://webfiles.york.ac.uk/OPENAIR/Anechoic/modern-viola/mono/) |
| `bassoon.wav` | Bassoon | 48 kHz, 16 bit, 39.2 s | not recorded yet |
| `female_speech.wav` | Female speech | 48 kHz, float, 13.3 s | [EARS dataset](https://sp-uhh.github.io/ears_dataset/) |
| `male_speech.wav` | Male speech | 48 kHz, float, 12.5 s | [EARS dataset](https://sp-uhh.github.io/ears_dataset/) |

The source and licence of `bassoon.wav` are not recorded yet. Fill them in here before the package is released.

`viola.wav` was resampled from 96 kHz to 48 kHz (polyphase, anti-aliased), the sampling rate of the package.

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
