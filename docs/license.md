# License and credits

## License

`compas_acoustics` is released under the MIT license.

```text
--8<-- "LICENSE"
```

## Authors

The authors are listed in [`CITATION.cff`](https://github.com/gramaziokohler/compas_acoustics/blob/main/CITATION.cff)
and in the citation on the [home page](index.md#how-to-cite), at Gramazio Kohler Research, ETH Zurich.

The `Binaural` and `Ambisonics` methods follow [hybrid-rir](https://github.com/gpapadimitri/hybrid-rir).

## Bundled data

The package ships data with its own sources and licences, which can be stricter than the MIT license of the code.
Check them before you use or distribute the package commercially.

| Data | Source | Licence |
|---|---|---|
| Materials library, 90 materials | The materials database of [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) | MIT (pyroomacoustics) |
| Materials library: `brickwall`, `carpet`, `concrete`, `acoustic_plaster`, `wall_absorber_50mm` | Unresolved: no source recorded | Unresolved |
| `KEMAR_FreeFieldComp_48kHz.sofa` (HRTF set of `Binaural`) | SONICOM HRTF dataset, Imperial College London. Cite: Engel et al. (2023), "The SONICOM HRTF Dataset", JAES 71(5), 241-253. | CC BY-SA 3.0. The share-alike condition applies to this data and its adaptations, not to the code. |
| `viola.wav` | [OpenAIR Anechoic Data](https://www.openair.hosted.york.ac.uk/?page_id=310), University of York | CC BY 4.0. Attribute www.openairlib.net / OpenAIR Anechoic Data. |
| `female_speech.wav`, `male_speech.wav` | [EARS dataset](https://sp-uhh.github.io/ears_dataset/). Cite: Richter et al. (2024), "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation", Interspeech 2024. | CC BY-NC 4.0: no commercial use. |
