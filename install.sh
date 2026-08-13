#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root (directory containing this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $0 [SUBCOMMAND] [OPTIONS]

One-click installer for ktransformers (sglang + kt-kernel).

This installer is designed for environments WITHOUT sudo/root access.
System/build dependencies are installed into the current Conda environment
using conda-forge.

SUBCOMMANDS:
  all             Full install: submodules → conda deps → sglang → kt-kernel (default)
  sglang          Install sglang only
  kt-kernel       Install kt-kernel only
  deps            Install build dependencies into current Conda environment only
  -h, --help      Show this help message

OPTIONS:
  --skip-sglang       Skip sglang installation (for "all" subcommand)
  --skip-kt-kernel    Skip kt-kernel installation (for "all" subcommand)
  --editable          Install sglang in editable/dev mode (-e)
  --manual            Pass through to kt-kernel (manual CPU config)
  --no-clean          Pass through to kt-kernel (skip build clean)

EXAMPLES:

  # Full install (recommended)
  $0

  # Install everything in editable mode for development
  $0 all --editable

  # Install sglang only
  $0 sglang

  # Install kt-kernel only
  $0 kt-kernel

  # Install kt-kernel only (manual CPU config)
  $0 kt-kernel --manual

  # Full install, skip sglang
  $0 all --skip-sglang

  # Install Conda dependencies only
  $0 deps

RECOMMENDED ENVIRONMENT:

  conda create -n ktransformers python=3.11 -y
  conda activate ktransformers
  $0

EOF
    exit 1
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

log_step() {
    echo ""
    echo "=========================================="
    echo "  $1"
    echo "=========================================="
    echo ""
}

log_info() {
    echo "[INFO] $1"
}

log_warn() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# Check that conda is available and, preferably, an environment is active.
check_conda() {
    if ! command -v conda >/dev/null 2>&1; then
        log_error "conda was not found in PATH."
        log_error ""
        log_error "This installer uses Conda instead of sudo/apt."
        log_error "Please install Miniconda/Miniforge first and activate an environment."
        log_error ""
        log_error "Example:"
        log_error "  conda create -n ktransformers python=3.11 -y"
        log_error "  conda activate ktransformers"
        log_error "  ./install.sh"
        exit 1
    fi

    if [ -z "${CONDA_PREFIX:-}" ]; then
        log_warn "CONDA_PREFIX is not set."
        log_warn "It looks like no Conda environment is currently activated."
        log_warn ""
        log_warn "It is strongly recommended to run:"
        log_warn "  conda create -n ktransformers python=3.11 -y"
        log_warn "  conda activate ktransformers"
        log_warn ""
        log_warn "Continuing with the Conda installation currently available in PATH."
    else
        log_info "Using Conda environment: $CONDA_PREFIX"
    fi
}

# Prefer python from the current environment.
get_python() {
    if command -v python >/dev/null 2>&1; then
        echo "python"
    elif command -v python3 >/dev/null 2>&1; then
        echo "python3"
    else
        log_error "Neither python nor python3 was found."
        exit 1
    fi
}

# Prefer python -m pip so pip and python always belong to the same environment.
pip_install() {
    local python_bin
    python_bin="$(get_python)"
    "$python_bin" -m pip install "$@"
}

# Read ktransformers version from version.py and export for sglang-kt.
read_kt_version() {
    local version_file="$REPO_ROOT/version.py"

    if [ ! -f "$version_file" ]; then
        log_warn "version.py not found; sglang-kt will use its default version"
        return 0
    fi

    local python_bin
    python_bin="$(get_python)"

    #
    # version.py has changed format across versions of ktransformers,
    # so try several common variable names.
    #
    KT_VERSION="$(
        VERSION_FILE="$version_file" "$python_bin" - <<'PY'
import os
import runpy

path = os.environ["VERSION_FILE"]
ns = runpy.run_path(path)

for key in ("__version__", "version", "VERSION"):
    value = ns.get(key)
    if value:
        print(value)
        break
PY
    )"

    if [ -n "${KT_VERSION:-}" ]; then
        export SGLANG_KT_VERSION="$KT_VERSION"
        log_info "ktransformers version: $KT_VERSION"
        log_info "SGLANG_KT_VERSION=$SGLANG_KT_VERSION"
    else
        log_warn "Could not determine version from version.py."
        log_warn "sglang-kt will use its default version."
    fi
}

# ─── Submodule init ───────────────────────────────────────────────────────────

init_submodules() {
    log_step "Initializing git submodules"

    #
    # .git can technically also be a file when using git worktree/submodules,
    # therefore use -e instead of only -d.
    #
    if [ ! -e "$REPO_ROOT/.git" ]; then
        log_warn "Not a git repository. Skipping submodule init."
        log_warn ""
        log_warn "If you need sglang, clone with:"
        log_warn "  git clone --recursive https://github.com/kvcache-ai/ktransformers.git"
        return 0
    fi

    cd "$REPO_ROOT"

    git submodule update --init --recursive

    log_info "Submodules initialized successfully."
}

# ─── Conda dependencies ───────────────────────────────────────────────────────

install_deps() {
    log_step "Installing build dependencies with Conda"

    check_conda

    log_info "No sudo or apt will be used."
    log_info "Installing dependencies from conda-forge..."
    log_info ""
    log_info "Packages:"
    log_info "  - cmake"
    log_info "  - libhwloc"
    log_info "  - pkg-config"
    log_info ""

    #
    # Equivalent user-space replacements for the packages that the upstream
    # kt-kernel installer installs with apt:
    #
    #   apt package        Conda package
    #   ----------------  -------------
    #   cmake              cmake
    #   libhwloc-dev       libhwloc
    #   pkg-config         pkg-config
    #
    # libhwloc from conda-forge provides the hwloc library and development files.
    #
    conda install -y -c conda-forge \
        cmake \
        libhwloc \
        pkg-config

    #
    # Make sure CMake/pkg-config can see libraries installed in CONDA_PREFIX.
    #
    if [ -n "${CONDA_PREFIX:-}" ]; then
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
        export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"
        export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"
        export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    log_info ""
    log_info "Conda dependencies installed successfully."

    if command -v cmake >/dev/null 2>&1; then
        log_info "cmake: $(command -v cmake)"
        cmake --version | head -n 1 || true
    fi

    if command -v pkg-config >/dev/null 2>&1; then
        log_info "pkg-config: $(command -v pkg-config)"
    fi

    #
    # Sanity check for hwloc.  The Conda package is named "libhwloc",
    # while the pkg-config module is still named "hwloc".
    #
    if command -v pkg-config >/dev/null 2>&1; then
        if pkg-config --exists hwloc; then
            log_info "hwloc: $(pkg-config --modversion hwloc)"
        else
            log_warn "pkg-config cannot currently find hwloc."
            log_warn "CMake may still find it through CMAKE_PREFIX_PATH."
        fi
    fi

    if [ -n "${CONDA_PREFIX:-}" ]; then
        if [ -f "$CONDA_PREFIX/include/hwloc.h" ]; then
            log_info "hwloc header: $CONDA_PREFIX/include/hwloc.h"
        else
            log_error "libhwloc was installed, but $CONDA_PREFIX/include/hwloc.h was not found."
            log_error "Please inspect the installed package with:"
            log_error "  conda list libhwloc"
            exit 1
        fi
    fi
}

# ─── sglang install ───────────────────────────────────────────────────────────

install_sglang() {
    local editable="${1:-0}"

    log_step "Installing sglang (kvcache-ai fork)"

    local sglang_dir="$REPO_ROOT/third_party/sglang"
    local pyproject="$sglang_dir/python/pyproject.toml"

    if [ ! -f "$pyproject" ]; then
        log_error "sglang source not found at:"
        log_error "  $sglang_dir"
        log_error ""
        log_error "Run:"
        log_error "  git submodule update --init --recursive"
        log_error ""
        log_error "or clone with:"
        log_error "  git clone --recursive https://github.com/kvcache-ai/ktransformers.git"
        exit 1
    fi

    cd "$sglang_dir"

    if [ "$editable" = "1" ]; then
        log_info "Installing sglang in editable mode..."
        pip_install -e "./python[all]"
    else
        log_info "Installing sglang..."
        pip_install "./python[all]"
    fi

    log_info "sglang installed successfully."
}

# ─── kt-kernel environment ────────────────────────────────────────────────────

setup_conda_build_env() {
    #
    # Ensure headers/libraries installed by Conda are visible to the kt-kernel
    # build even though upstream's install.sh expects system packages.
    #
    if [ -n "${CONDA_PREFIX:-}" ]; then
        export CMAKE_PREFIX_PATH="${CONDA_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

        export PKG_CONFIG_PATH="${CONDA_PREFIX}/lib/pkgconfig:${CONDA_PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

        export CPATH="${CONDA_PREFIX}/include${CPATH:+:${CPATH}}"

        export C_INCLUDE_PATH="${CONDA_PREFIX}/include${C_INCLUDE_PATH:+:${C_INCLUDE_PATH}}"
        export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

        export LIBRARY_PATH="${CONDA_PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"

        export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

        #
        # This is useful for CMake find_package()/find_library() calls.
        #
        export CMAKE_LIBRARY_PATH="${CONDA_PREFIX}/lib${CMAKE_LIBRARY_PATH:+:${CMAKE_LIBRARY_PATH}}"
        export CMAKE_INCLUDE_PATH="${CONDA_PREFIX}/include${CMAKE_INCLUDE_PATH:+:${CMAKE_INCLUDE_PATH}}"

        log_info "Using Conda libraries from: $CONDA_PREFIX"
    fi
}

# ─── kt-kernel install ────────────────────────────────────────────────────────

install_kt_kernel() {
    # Forward all remaining args to kt-kernel/install.sh build.
    local kt_args=("$@")

    log_step "Installing kt-kernel"

    local kt_install="$REPO_ROOT/kt-kernel/install.sh"

    if [ ! -f "$kt_install" ]; then
        log_error "kt-kernel/install.sh not found at:"
        log_error "  $kt_install"
        exit 1
    fi

    check_conda
    setup_conda_build_env

    #
    # IMPORTANT:
    #
    # DO NOT call:
    #
    #   bash ./install.sh deps
    #   bash ./install.sh all
    #
    # because the upstream kt-kernel dependency installer invokes apt/dnf/etc.
    #
    # We already installed cmake/hwloc/pkg-config through Conda above.
    #
    # The "build" subcommand only builds kt-kernel and does not install
    # system dependencies.
    #
    cd "$REPO_ROOT/kt-kernel"

    log_info "Building kt-kernel without running its system dependency installer..."

    bash ./install.sh build "${kt_args[@]}"

    log_info "kt-kernel installed successfully."
}

# ─── "all" subcommand ─────────────────────────────────────────────────────────

install_all() {
    local skip_sglang=0
    local skip_kt_kernel=0
    local editable=0
    local kt_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --skip-sglang)
                skip_sglang=1
                shift
                ;;

            --skip-kt-kernel)
                skip_kt_kernel=1
                shift
                ;;

            --editable)
                editable=1
                shift
                ;;

            --manual)
                kt_args+=("--manual")
                shift
                ;;

            --no-clean)
                kt_args+=("--no-clean")
                shift
                ;;

            -h|--help)
                usage
                ;;

            *)
                log_error "Unknown option: $1"
                usage
                ;;
        esac
    done

    # 0. Make sure Conda is available
    check_conda

    # 1. Init submodules
    init_submodules

    # 2. Install dependencies through Conda instead of sudo apt
    install_deps

    # 3. Read version for sglang-kt
    read_kt_version

    # 4. Install sglang
    if [ "$skip_sglang" = "0" ]; then
        install_sglang "$editable"
    else
        log_info "Skipping sglang installation (--skip-sglang)."
    fi

    # 5. Build & install kt-kernel
    if [ "$skip_kt_kernel" = "0" ]; then
        install_kt_kernel "${kt_args[@]}"
    else
        log_info "Skipping kt-kernel installation (--skip-kt-kernel)."
    fi

    log_step "Installation complete!"

    echo "Conda environment:"
    echo "  ${CONDA_PREFIX:-unknown}"
    echo ""

    echo "Verify with:"
    echo "  kt doctor"
    echo ""

    echo "Or:"
    echo "  kt version"
    echo ""
}

# ─── Subcommand dispatcher ────────────────────────────────────────────────────

SUBCMD="all"

if [[ $# -gt 0 ]]; then
    case "$1" in
        all|sglang|kt-kernel|deps)
            SUBCMD="$1"
            shift
            ;;

        -h|--help)
            usage
            ;;

        -*)
            # Flags without subcommand → default to "all"
            SUBCMD="all"
            ;;

        *)
            log_error "Unknown subcommand: $1"
            usage
            ;;
    esac
fi

case "$SUBCMD" in
    all)
        install_all "$@"
        ;;

    sglang)
        # Parse sglang-specific options
        editable=0

        while [[ $# -gt 0 ]]; do
            case "$1" in
                --editable)
                    editable=1
                    shift
                    ;;

                -h|--help)
                    usage
                    ;;

                *)
                    log_error "Unknown option for sglang: $1"
                    usage
                    ;;
            esac
        done

        check_conda
        init_submodules
        read_kt_version
        install_sglang "$editable"
        ;;

    kt-kernel)
        #
        # kt-kernel standalone installation should also install the Conda
        # dependencies first, otherwise hwloc/pkg-config may be missing.
        #
        check_conda
        init_submodules
        install_deps
        install_kt_kernel "$@"
        ;;

    deps)
        install_deps
        ;;
esac