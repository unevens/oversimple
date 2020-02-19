# oversimple

Oversimple is a C++17 library for audio oversampling, which tries to offer a simple api.

Oversimple wraps two of the best resampling libraries available: 

- HIIR, by [Laurent De Soras](http://ldesoras.free.fr/) for minimum phase antialiasing, through [my fork](https://github.com/unevens/hiir) which adds support for double precision floating-point numbers, and AVX instructions. HIIR only implements power of two resampling.

- and [r8brain-free-src](https://github.com/avaneev/r8brain-free-src) by Aleksey Vaneev, for linear phase antialiasing, through [my fork](https://github.com/unevens/r8brain/tree/include), which adds supports for [my fork of Julien Pommier's PFFFT library](https://github.com/unevens/pffft), which can work with double-precision floating point numbers using AVX instructions.

Aligned memory and interleaved buffers needed by the simd code in HIIR are managed using [avec](https://github.com/unevens/avec).

## Usage

Add everything to your project except for the content of the `test` folders and the .cpp files in `avec/vectorclass`.

Add to your include paths the directory in which you put this repository and its subdirectories `r8brain`, `avec`, and `avec/vectorclass`. 

If you support AVX instructions, you may want to define `R8B_PFFFT_DOUBLE=1` in `r8brain/r8bconf.h` or as a preprocessor definition. See `r8brain/README.md` for more details.

## Dependencies

- [Boost.Align](https://www.boost.org/doc/libs/1_71_0/doc/html/align.html).
- `pthread` on *nix (only r8brain).

## Documentation

The documentation, available at https://unevens.github.io/oversimple/, can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```
