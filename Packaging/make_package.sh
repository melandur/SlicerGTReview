#!/usr/bin/env bash
#
# Build an extension package that Slicer's Extensions Manager will accept
# through "Install from file", for Linux, macOS or Windows.
#
# No Slicer build tree is needed.  GTReview is python-only, and the extensions
# manager asks for very little: an archive with a single top-level directory
# holding an .s4ext description file.  The name of that file becomes the
# extension name, and the directory's contents become the installed tree.
# (Slicer/Base/QTCore/qSlicerExtensionsManagerModel.cxx -- installExtension()
# scans the archive for *.s4ext, extractExtensionArchive() requires the single
# top-level directory.)
#
# Two things make a package platform-specific:
#
#   * The module is installed under lib/Slicer-<major.minor>/qt-scripted-modules
#     and Slicer looks nowhere else, so a package built for 5.10 is invisible to
#     5.12.  That directory name is read from the Slicer install passed in.
#
#   * macOS wants one more level of nesting.  extractExtensionArchive copies
#     from <archive>/Slicer.app/Contents/Extensions-<revision>/<name> there,
#     against <archive>/ on Linux and Windows (the Slicer_OS_MAC_NAME branch,
#     using Slicer_BUNDLE_LOCATION = "Slicer.app/Contents").  The revision in
#     that path is the RUNNING Slicer's, so a macOS package is tied to one
#     revision and not merely one minor version -- pass --revision to match the
#     machine it is for.
#
# Usage:
#   Packaging/make_package.sh [--slicer <dir>] [--os linux|macosx|win]
#                             [--revision <rev>] [--slicer-version <x.y>]
#                             [--output <dir>]
#
#   --slicer          a Slicer installation to read the layout from
#   --os              target platform (default: detected from --slicer, else this host)
#   --revision        target Slicer revision, e.g. 34045.  Cosmetic on Linux and
#                     Windows, LOAD-BEARING on macOS (see above)
#   --slicer-version  major.minor, e.g. 5.10, when building without a --slicer
#   --output          where to write the archive (default: the repository root)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLICER_DIR="${SLICER_HOME:-}"
OUTPUT_DIR="$REPO"
TARGET_OS=""
REVISION=""
SLICER_MINOR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --slicer) SLICER_DIR="$2"; shift 2 ;;
        --os) TARGET_OS="$2"; shift 2 ;;
        --revision) REVISION="$2"; shift 2 ;;
        --slicer-version) SLICER_MINOR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Where things live differs per platform, so find the installation first and
# let it answer the questions rather than guessing from the host.
# ---------------------------------------------------------------------------
if [ -z "$SLICER_DIR" ]; then
    SLICER_DIR="$(ls -d "$HOME"/Documents/Slicer-* "$HOME"/Slicer-* \
                      /Applications/Slicer.app 2>/dev/null | sort -V | tail -1 || true)"
fi

detect_os() {
    local dir="$1"
    if [ -x "$dir/Contents/MacOS/Slicer" ] || [ -d "$dir/Contents/MacOS" ]; then
        echo macosx
    elif [ -f "$dir/Slicer.exe" ]; then
        echo win
    elif [ -x "$dir/Slicer" ]; then
        echo linux
    fi
}

if [ -n "$SLICER_DIR" ] && [ -d "$SLICER_DIR" ]; then
    [ -n "$TARGET_OS" ] || TARGET_OS="$(detect_os "$SLICER_DIR")"
fi
if [ -z "$TARGET_OS" ]; then
    case "$(uname -s)" in
        Darwin) TARGET_OS=macosx ;;
        MINGW*|MSYS*|CYGWIN*) TARGET_OS=win ;;
        *) TARGET_OS=linux ;;
    esac
fi
case "$TARGET_OS" in
    linux|macosx|win) ;;
    mac|osx|darwin) TARGET_OS=macosx ;;
    windows) TARGET_OS=win ;;
    *) echo "error: --os must be linux, macosx or win (got '$TARGET_OS')" >&2; exit 2 ;;
esac

# On macOS the installation IS the bundle, so lib/ sits inside Contents/.
INSTALL_PREFIX="$SLICER_DIR"
[ "$TARGET_OS" = macosx ] && [ -d "$SLICER_DIR/Contents" ] && INSTALL_PREFIX="$SLICER_DIR/Contents"

if [ -z "$SLICER_MINOR" ]; then
    LIB_DIR="$(ls -d "$INSTALL_PREFIX"/lib/Slicer-* 2>/dev/null | head -1 || true)"
    if [ -z "$LIB_DIR" ]; then
        echo "error: could not find lib/Slicer-<major.minor> under $INSTALL_PREFIX." >&2
        echo "       pass --slicer <installation> or --slicer-version <x.y>" >&2
        exit 1
    fi
    SLICER_MINOR="$(basename "$LIB_DIR" | sed 's/^Slicer-//')"
fi

if [ -z "$REVISION" ]; then
    REVISION="$(ls -d "$INSTALL_PREFIX"/slicer.org/Extensions-* "$INSTALL_PREFIX"/Extensions-* \
                    2>/dev/null | head -1 | sed 's/.*Extensions-//' || true)"
fi
if [ -z "$REVISION" ]; then
    if [ "$TARGET_OS" = macosx ]; then
        echo "error: macOS packages embed the target Slicer's revision in their" >&2
        echo "       directory layout and it cannot be guessed; pass --revision" >&2
        exit 1
    fi
    REVISION="rev"
fi

VERSION="$(cd "$REPO" && git describe --tags --always --dirty 2>/dev/null || echo "untagged")"

# ---------------------------------------------------------------------------
# Metadata comes from the top-level CMakeLists.txt so there is one source of
# truth: the .s4ext below and a real CPack build must not disagree.
# ---------------------------------------------------------------------------
cmake_value() {
    sed -n "s/^set($1 \"\{0,1\}\(.*\)/\1/p" "$REPO/CMakeLists.txt" |
        head -1 | sed 's/) *#.*$//; s/)$//; s/"$//'
}
EXTENSION_NAME="$(sed -n 's/^set(EXTENSION_NAME \(.*\))$/\1/p' "$REPO/CMakeLists.txt" | head -1)"
[ -n "$EXTENSION_NAME" ] || { echo "error: EXTENSION_NAME not found in CMakeLists.txt" >&2; exit 1; }

HOMEPAGE="$(cmake_value EXTENSION_HOMEPAGE)"
CATEGORY="$(cmake_value EXTENSION_CATEGORY)"
CONTRIBUTORS="$(cmake_value EXTENSION_CONTRIBUTORS)"
DESCRIPTION="$(cmake_value EXTENSION_DESCRIPTION)"
ICONURL="$(cmake_value EXTENSION_ICONURL)"
SCREENSHOTURLS="$(cmake_value EXTENSION_SCREENSHOTURLS)"
STATUS="$(cmake_value EXTENSION_STATUS)"
DEPENDS="$(cmake_value EXTENSION_DEPENDS)"
BUILD_SUBDIRECTORY="$(cmake_value EXTENSION_BUILD_SUBDIRECTORY)"
SCM_REVISION="$(cd "$REPO" && git rev-parse HEAD 2>/dev/null || echo NA)"

# ---------------------------------------------------------------------------
# The manifest is authoritative, and a file missing from it is the failure mode
# the module CMakeLists warns about: it keeps working from the source tree and
# only breaks once packaged.  Refuse to build rather than ship that.
# ---------------------------------------------------------------------------
MODULE_DIR="$REPO/$EXTENSION_NAME"
mapfile -t SCRIPTS < <(
    sed -n '/^set(MODULE_PYTHON_SCRIPTS/,/^  )/p' "$MODULE_DIR/CMakeLists.txt" |
        sed -n 's/^  \${MODULE_NAME}\(.*\)$/'"$EXTENSION_NAME"'\1/p'
)
[ "${#SCRIPTS[@]}" -gt 0 ] || { echo "error: MODULE_PYTHON_SCRIPTS is empty" >&2; exit 1; }

missing=0
for script in "${SCRIPTS[@]}"; do
    [ -f "$MODULE_DIR/$script" ] || { echo "error: listed but absent: $script" >&2; missing=1; }
done
while IFS= read -r found; do
    printf '%s\n' "${SCRIPTS[@]}" | grep -qxF "$found" ||
        { echo "error: on disk but not in MODULE_PYTHON_SCRIPTS: $found" >&2; missing=1; }
done < <(cd "$MODULE_DIR" && find . -name '*.py' -not -path './__pycache__/*' \
            -not -path '*/__pycache__/*' | sed 's|^\./||' | sort)
[ "$missing" -eq 0 ] || exit 1

RESOURCES=(Resources/Icons/"$EXTENSION_NAME".png)

# ---------------------------------------------------------------------------
# Assemble.  INNER is what the manager copies into place; on macOS it is buried
# under the bundle path the mac branch of extractExtensionArchive looks for.
# ---------------------------------------------------------------------------
ARCHIVE_BASE="${REVISION}-${TARGET_OS}-amd64-${EXTENSION_NAME}-${VERSION}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/$ARCHIVE_BASE"
if [ "$TARGET_OS" = macosx ]; then
    INNER="$ROOT/Slicer.app/Contents/Extensions-$REVISION/$EXTENSION_NAME"
else
    INNER="$ROOT"
fi
SCRIPTED="$INNER/lib/Slicer-$SLICER_MINOR/qt-scripted-modules"
SHARE="$INNER/share/Slicer-$SLICER_MINOR"
mkdir -p "$SCRIPTED" "$SHARE"

for script in "${SCRIPTS[@]}"; do
    mkdir -p "$SCRIPTED/$(dirname "$script")"
    cp "$MODULE_DIR/$script" "$SCRIPTED/$script"
done
for resource in "${RESOURCES[@]}"; do
    mkdir -p "$SCRIPTED/$(dirname "$resource")"
    cp "$MODULE_DIR/$resource" "$SCRIPTED/$resource"
done

# Key-value pairs, first token is the key, '#' starts a comment.  Parsed by
# qSlicerExtensionsManagerModel::parseExtensionDescriptionFile.
cat > "$SHARE/$EXTENSION_NAME.s4ext" <<EOF
# Generated by Packaging/make_package.sh from the top-level CMakeLists.txt.
# The extension name is taken from THIS FILE'S NAME, not from any key here.
scm git
scmurl $HOMEPAGE
scmrevision $SCM_REVISION
depends $DEPENDS
build_subdirectory $BUILD_SUBDIRECTORY
homepage $HOMEPAGE
contributors $CONTRIBUTORS
category $CATEGORY
iconurl $ICONURL
description $DESCRIPTION
screenshoturls $SCREENSHOTURLS
status $STATUS
enabled 1
EOF

mkdir -p "$OUTPUT_DIR"
ARCHIVE="$OUTPUT_DIR/$ARCHIVE_BASE.tar.gz"
tar -czf "$ARCHIVE" -C "$STAGE" "$ARCHIVE_BASE"

echo "built $ARCHIVE"
echo
echo "  extension   $EXTENSION_NAME $VERSION"
echo "  target      $TARGET_OS, Slicer $SLICER_MINOR, revision $REVISION"
echo "  files       ${#SCRIPTS[@]} scripts + ${#RESOURCES[@]} resource(s)"
if [ "$TARGET_OS" = macosx ]; then
    echo "  note        macOS packages are tied to revision $REVISION, not just to $SLICER_MINOR"
else
    echo "  note        a package for Slicer $SLICER_MINOR is invisible to any other minor version"
fi
echo
echo "Install it: Slicer -> View -> Extensions Manager -> Install from file,"
echo "pick this archive, then restart.  GT Review appears under Segmentation."
