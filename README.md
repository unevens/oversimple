# oversimple

Oversimple is a C++17 library for audio oversampling, which tries to offer a simple api.

Oversimple wraps two of the best resampling libraries available: 

- [HIIR](https://github.com/unevens/hiir), by [Laurent De Soras](http://ldesoras.free.fr/) for minimum phase antialiasing. HIIR only implements power of two resampling.

- and [r8brain-free-src](https://github.com/avaneev/r8brain-free-src) by Aleksey Vaneev, for linear phase antialiasing, through [my fork](https://github.com/unevens/r8brain/tree/with-pffft-double), which adds supports for the latest [PFFFT implementation](https://github.com/marton78/pffft) which uses SIMD instructions for both single and double precision floating point numbers on all platforms where they are supported (including double precision on ARM AArch64).

Aligned memory and interleaved buffers needed by the simd code in HIIR are managed using [avec](https://github.com/unevens/avec).

## Usage

Add everything to your project except for the content of the `test` folders and the .cpp files in `avec/vectorclass`.

Add to your include paths the directory in which you put this repository and its subdirectories `r8brain`, `avec`, and `avec/vectorclass`. 

To use PFFFT with double precision, define `R8B_PFFFT_DOUBLE=1` in `r8brain/r8bconf.h` or as a preprocessor definition. See `r8brain/README.md` for more details.

## Dependencies

- `pthread` on *nix (only r8brain).

## Documentation

The documentation, available at https://unevens.github.io/oversimple/, can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```
