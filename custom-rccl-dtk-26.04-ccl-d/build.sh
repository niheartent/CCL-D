#!/bin/bash

# 设置默认值
ROCM_PATH="${ROCM_PATH:-}"
BUILD_TESTS="ON"
FULL_STRESS_TEST="OFF"

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --rocm-path PATH    Set ROCm installation path (default: \$ROCM_PATH)"
    echo "  -t, --build-tests ON|OFF Enable/disable test building (default: ON)"
    echo "  -f, --full-stress ON|OFF Enable/disable full stress test when tests are enabled (default: OFF)"
    echo "  -h, --help             Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--rocm-path)
            ROCM_PATH="$2"
            shift 2
            ;;
        -t|--build-tests)
            BUILD_TESTS="$2"
            shift 2
            ;;
        -f|--full-stress)
            FULL_STRESS_TEST="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ -z "$ROCM_PATH" ]]; then
    echo "Error: ROCM_PATH is not set and not specified via command line"
    show_help
    exit 1
fi

if [[ ! -d "$ROCM_PATH" ]]; then
    echo "Error: ROCm path does not exist: $ROCM_PATH"
    exit 1
fi

if [[ "$BUILD_TESTS" != "ON" && "$BUILD_TESTS" != "OFF" ]]; then
    echo "Error: BUILD_TESTS must be ON or OFF"
    exit 1
fi

if [[ "$FULL_STRESS_TEST" != "ON" && "$FULL_STRESS_TEST" != "OFF" ]]; then
    echo "Error: FULL_STRESS_TEST must be ON or OFF"
    exit 1
fi

if [[ "$BUILD_TESTS" == "OFF" ]]; then
    FULL_STRESS_TEST="OFF"
fi

# 创建构建目录（如果不存在）
mkdir -p build
cd build

# 执行CMake配置
echo "Configuring with:"
echo "  ROCM_PATH=$ROCM_PATH"
echo "  BUILD_TESTS=$BUILD_TESTS"
echo "  FULL_STRESS_TEST=$FULL_STRESS_TEST"

cmake .. \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}" \
    -DBUILD_TESTS="${BUILD_TESTS}" \
    -DFULL_STRESS_TEST="${FULL_STRESS_TEST}"

# 编译
echo "Starting build..."
make -j$(nproc)