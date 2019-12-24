# oversimple

Oversimple is a library for power-of-two audio resampling, which tries to offer a simple api.

Oversimple wraps two of the best resampling libraries available: 

- HIIR, by [Laurent De Soras](http://ldesoras.free.fr/) for minimum phase antialiasing, through [my fork](https://github.com/unevens/hiir) which adds support for double precision floating-point numbers, and AVX instructions.

- and [r8brain-free-src](https://github.com/avaneev/r8brain-free-src) by Aleksey Vaneev, for linear phase antialiasing, through [my fork](https://github.com/unevens/r8brain/tree/include), which adds supports for [my fork of Julien Pommier's PFFFT library](https://github.com/unevens/pffft), which can work with double-precision floating point numbers using AVX instructions.

Aligned memory and interleaved buffers needed by the simd code in HIIR are managed using my header only library [avec](https://github.com/unevens/avec).

## Usage

Just add everything to your project except for the content of the `test` folders and the .cpp files in `audio-vec/vectorclass`.

If you support AVX instructions, you may want to define `R8B_PFFFT_DOUBLE=1` in `r8bconf.h` or as a preprocessor definition. See `r8brain/README.md` for more details.

## Dependencies

- [Boost.Align](https://www.boost.org/doc/libs/1_71_0/doc/html/align.html).
- `pthread` on *nix (only r8brain).

## Documentation

The documentation can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```
