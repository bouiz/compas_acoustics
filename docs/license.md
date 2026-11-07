# License and credits

## License

`compas_acoustics` is released under the MIT license.

```text
--8<-- "LICENSE"
```

## Authors

The authors are listed in [`CITATION.cff`](https://github.com/gramaziokohler/compas_acoustics/blob/main/CITATION.cff)
and in the citation on the [home page](index.md#how-to-cite), at Gramazio Kohler Research, ETH Zurich.

## Bundled data

The package ships data with its own sources and licences, which can be stricter than the MIT license of the code.
Check them before you use or distribute the package commercially.

| Data | Source | Licence |
|---|---|---|
| Materials library, 90 materials | The materials database of [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) | MIT (pyroomacoustics) |
| Materials library: `brickwall`, `carpet`, `concrete`, `acoustic_plaster`, `wall_absorber_50mm` | Unresolved: no source recorded | Unresolved |
| `KEMAR_FreeFieldComp_48kHz.sofa` (HRTF set of `Binaural`) | SONICOM HRTF dataset, Imperial College London. Cite: Engel et al. (2023), "The SONICOM HRTF Dataset", JAES 71(5), 241-253. | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). The share-alike condition applies to this data and its adaptations, not to the code. |
| `viola.wav` | [OpenAIR Anechoic Data](https://www.openair.hosted.york.ac.uk/?page_id=310), University of York | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute www.openairlib.net / OpenAIR Anechoic Data. |
| `female_speech.wav`, `male_speech.wav` | [EARS dataset](https://sp-uhh.github.io/ears_dataset/). Cite: Richter et al. (2024), "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation", Interspeech 2024. | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): no commercial use. |

## Third-party code

Parts of the code are adapted from other projects, under their own licences.

**Edge diffraction** (`compas_acoustics.hybrid.btms`): `wedge1st_ir`, `wedge2nd_ir` and `_coordtrans1` follow
`EDwedge1st_ir`, `EDwedge2nd` and `EDcoordtrans1` of the
[Edge Diffraction Toolbox](https://github.com/upsvensson/Edge-diffraction-Matlab-toolbox) v0.504 by Peter Svensson,
under the BSD license:

```text
Copyright (c) 2018, Peter Svensson All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDEN- TAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUD- ING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTER- RUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

**Air** (`compas_acoustics.room._pyfar_constants`): Cramer's speed of sound and the ISO 9613-1 air attenuation are
copied from [pyfar](https://github.com/pyfar/pyfar) 0.8.1 (`pyfar.constants`), for Python versions that pyfar 0.8
does not support, under the MIT license:

```text
Copyright (c) [2020] [The pyfar developers]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
