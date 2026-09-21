#!/usr/bin/env bash
# 用 arduino-cli 编译固件，不打开 Arduino IDE。
#
# 用法:
#   ./tools/build.sh                      # 自动找 arduino-cli
#   ARDUINO_CLI=/path/to/arduino-cli ./tools/build.sh
#   FQBN=esp32:esp32:esp32s3 ./tools/build.sh
#
# 为什么需要这个脚本而不是直接点 IDE 里的「验证」：
#   · CI / 命令行下可复现，别人 clone 下来就能验证固件是否编译得过
#   · 两个 Windows 特有的坑（见下面的 to_native / BUILD_DIR）
set -euo pipefail

# arduino-cli 是 Windows 原生可执行文件，不认 MSYS 形式的路径。
# 传 "/d/foo" 进去，它会把盘符和路径拼成 "D:\d\foo"，报 "Can't open sketch"。
# cygpath -m 转成 "D:/foo"（正斜杠，Windows 程序也接受）。
to_native() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

SKETCH="$(cd "$(dirname "$0")/.." && pwd)/firmware/HomeGuard"
SKETCH="$(to_native "$SKETCH")"
FQBN="${FQBN:-esp32:esp32:esp32s3}"

# 构建目录：GNU ld 在含非 ASCII 字符（比如中文）的路径下会报
# "cannot open output file ... No such file or directory"。
# 默认放到临时目录下的 ASCII 路径，避开用户目录里的中文。
# 注意 sketch 路径本身可以是中文的，只有输出目录不行。
BUILD_DIR="${BUILD_DIR:-$(mktemp -d)}"
BUILD_DIR="$(to_native "$BUILD_DIR")"

# 找 arduino-cli：环境变量 > PATH > Arduino IDE 自带的那一份
if [ -z "${ARDUINO_CLI:-}" ]; then
  if command -v arduino-cli >/dev/null 2>&1; then
    ARDUINO_CLI="$(command -v arduino-cli)"
  else
    for c in \
      "$(to_native "/d/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe")" \
      "$(to_native "/c/Program Files/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe")" \
      "$LOCALAPPDATA/Programs/Arduino IDE/resources/app/lib/backend/resources/arduino-cli.exe"
    do
      if [ -x "$c" ]; then
        ARDUINO_CLI="$c"
        break
      fi
    done
  fi
fi

if [ -z "${ARDUINO_CLI:-}" ] || [ ! -x "${ARDUINO_CLI:-}" ]; then
  echo "找不到 arduino-cli。请用 ARDUINO_CLI=/path/to/arduino-cli 指定。" >&2
  exit 127
fi

echo "arduino-cli : $ARDUINO_CLI"
echo "FQBN        : $FQBN"
echo "sketch      : $SKETCH"
echo "build dir   : $BUILD_DIR"
echo

"$ARDUINO_CLI" compile --fqbn "$FQBN" --build-path "$BUILD_DIR" "$SKETCH"
