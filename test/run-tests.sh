#!/usr/bin/env bash
# Build the test binary for every native + cross-architecture combo this
# host supports, run each, and print a pass/fail summary.
#
# macOS Apple Silicon:  builds + runs arm64 native, plus x86_64 via Rosetta
#                       (skipped automatically if Rosetta isn't installed).
# macOS Intel:          builds + runs x86_64.
# Linux:                builds + runs the host arch.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Project name = the parent directory of test/ (avec or oversimple). The
# CMakeLists creates an executable named "${PROJECT_NAME}-test".
PROJECT_NAME="$(basename "$(dirname "$SCRIPT_DIR")")"

HOST_ARCH=$(uname -m)
PLATFORM=$(uname)

# Architectures to build for.
ARCHS=()
case "$PLATFORM" in
  Darwin)
    if [[ "$HOST_ARCH" == "arm64" ]]; then
      ARCHS=("arm64" "x86_64")
    else
      ARCHS=("$HOST_ARCH")
    fi
    ;;
  Linux)
    ARCHS=("$HOST_ARCH")
    ;;
  *)
    echo "unsupported platform: $PLATFORM" >&2
    exit 1
    ;;
esac

# Prefer Homebrew Clang on macOS for parity with the plugin builds (which
# also use it because of the upstream Apple Clang NEON codegen bug).
CC_ARG=""; CXX_ARG=""
if [[ "$PLATFORM" == "Darwin" && -x /opt/homebrew/opt/llvm/bin/clang++ ]]; then
  CC_ARG="-DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang"
  CXX_ARG="-DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++"
fi

if command -v ninja >/dev/null 2>&1; then
  GENERATOR=Ninja
else
  GENERATOR="Unix Makefiles"
fi

results=()
fails=0

# Detect Rosetta once. `arch -x86_64 true` on Apple Silicon without Rosetta
# fails with "Bad CPU type in executable"; with Rosetta it just returns 0.
have_rosetta=1
if [[ "$PLATFORM" == "Darwin" && "$HOST_ARCH" == "arm64" ]]; then
  if ! arch -x86_64 /usr/bin/true 2>/dev/null; then
    have_rosetta=0
  fi
fi

for arch in "${ARCHS[@]}"; do
  echo
  echo "============================================================"
  echo "=== Building $PROJECT_NAME tests for $arch"
  echo "============================================================"
  BUILD_DIR="build/$arch"
  rm -rf "$BUILD_DIR"
  cmake_args=(-S "$SCRIPT_DIR" -B "$BUILD_DIR" -G "$GENERATOR"
              -DCMAKE_BUILD_TYPE=Release)
  [[ -n "$CC_ARG"  ]] && cmake_args+=("$CC_ARG" "$CXX_ARG")
  if [[ "$PLATFORM" == "Darwin" ]]; then
    cmake_args+=("-DCMAKE_OSX_ARCHITECTURES=$arch")
  fi

  if ! cmake "${cmake_args[@]}"; then
    results+=("$arch: CONFIGURE FAILED")
    fails=$((fails+1))
    continue
  fi
  if ! cmake --build "$BUILD_DIR" --parallel; then
    results+=("$arch: BUILD FAILED")
    fails=$((fails+1))
    continue
  fi

  bin="$BUILD_DIR/${PROJECT_NAME}-test"
  if [[ ! -x "$bin" ]]; then
    results+=("$arch: BINARY NOT FOUND at $bin")
    fails=$((fails+1))
    continue
  fi

  echo
  echo "=== Running $PROJECT_NAME tests for $arch ==="
  if [[ "$PLATFORM" == "Darwin" && "$arch" != "$HOST_ARCH" ]]; then
    if [[ "$have_rosetta" -eq 0 ]]; then
      results+=("$arch: BUILT but NOT RUN (Rosetta missing)")
      continue
    fi
    if arch -"$arch" "$bin"; then
      results+=("$arch: PASS")
    else
      rc=$?
      results+=("$arch: FAIL (exit $rc)")
      fails=$((fails+1))
    fi
  else
    if "$bin"; then
      results+=("$arch: PASS")
    else
      rc=$?
      results+=("$arch: FAIL (exit $rc)")
      fails=$((fails+1))
    fi
  fi
done

echo
echo "============================================================"
echo "=== Summary ($PROJECT_NAME)"
echo "============================================================"
for r in "${results[@]}"; do echo "  $r"; done
echo
if [[ $fails -eq 0 ]]; then
  echo "ALL ARCHES PASSED"
  exit 0
else
  echo "$fails ARCH(S) FAILED"
  exit 1
fi
