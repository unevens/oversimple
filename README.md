# oversimple

Oversimple is a C++17 library for audio oversampling, which tries to offer a simple api.

Oversimple wraps two of the best resampling libraries available: 

- [HIIR](https://github.com/unevens/hiir), by [Laurent De Soras](http://ldesoras.free.fr/) for minimum phase antialiasing. HIIR only implements power of two resampling.

- and [r8brain-free-src](https://github.com/avaneev/r8brain-free-src) by Aleksey Vaneev, for linear phase antialiasing.

Both libraries use SIMD instructions for both single and double precision floating point numbers on all platforms where they are supported (including double precision on ARM AArch64).

Aligned memory and interleaved buffers needed by the simd code in HIIR are managed using [avec](https://github.com/unevens/avec).

## Usage

Add everything to your project except for the content of the `test` folders and the .cpp files in `avec/vectorclass`.

Add to your include paths the directory in which you put this repository and its subdirectories `r8brain`, `avec`, and `avec/vectorclass`. 

To use PFFFT with double precision, define `R8B_PFFFT_DOUBLE=1` in `r8brain/r8bconf.h` or as a preprocessor definition. See `r8brain/README.md` for more details.

## Dependencies

- `pthread` on *nix (only r8brain).

## Tests

```bash
$ cd test
$ ./run-tests.sh
```

builds and runs the doctest-based test suite for every architecture this host
supports. On Apple Silicon that's both `arm64` and `x86_64` (via Rosetta);
on Intel and Linux it's just the host arch.

The suite measures round-trip SNR per channel for each oversampler (FIR
linear-phase, IIR minimum-phase, the `Oversampling` wrapper) and asserts
against conservative per-test thresholds (≥60 dB steady-state for FIR
sine and `Oversampling` linear-phase sine, ≥60 dB IIR DC convergence,
permissive first-block thresholds during FIR warm-up). The previous
test only printed SNR to stdout without asserting — a 50 dB regression
would have been invisible.

Both float and double precisions are tested for every (test, arch) pair.

## Documentation

The documentation, available at https://unevens.github.io/oversimple/, can be generated with [Doxygen](http://doxygen.nl/) running

```bash
$ doxygen doxyfile.txt
```
