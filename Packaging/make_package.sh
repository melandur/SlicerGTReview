#!/usr/bin/env bash
#
# Build an extension package that Slicer's Extensions Manager will accept
# through "Install from file".
#
# No Slicer build tree is needed.  GTReview is python-only, and the extensions
# manager asks for very little: an archive with a single top-level directory,
# holding an .s4ext description file anywhere inside it.  The name of that file
# becomes the extension name, and the directory's contents become the installed
# tree.  (Slicer/Base/QTCore/qSlicerExtensionsManagerModel.cxx --
# installExtension() scans the archive for *.s4ext, extractExtensionArchive()
# requires the single top-level directory.)
#
# The version-specific part is lib/Slicer-<major.minor>/qt-scripted-modules:
# Slicer only looks there, so a package built for 5.10 is invisible to 5.12.
# That directory name is read from the Slicer install passed in, which is why
# this script wants one rather than guessing.
#
# Usage:
#   Packaging/make_package.sh [--slicer <slicer-install-dir>] [--output <dir>]
#
#   --slicer   a Slicer installation, e.g. ~/Documents/Slicer-5.10.0-linux-amd64
#              (default: $SLICER_HOME, else the newest ~/Documents/Slicer-*)
#   --output   where to write the .tar.gz (default: the repository root)
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLICER_DIR="${SLICER_HOME:-}"
OUTPUT_DIR="$REPO"

while [ $# -gt 0 ]; do
    case "$1" in
        --slicer) SLICER_DIR="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$SLICER_DIR" ]; then
    SLICER_DIR="$(ls -d "$HOME"/Documents/Slicer-*-linux-amd64 2>/dev/null | sort -V | tail -1 || true)"
fi
if [ -z "$SLICER_DIR" ] || [ ! -x "$SLICER_DIR/Slicer" ]; then
    echo "error: no Slicer installation found; pass --slicer <dir>" >&2
    exit 1
fi

# lib/Slicer-5.10 -> 5.10.  Everything installed has to sit under that name.
LIB_DIR="$(ls -d "$SLICER_DIR"/lib/Slicer-* 2>/dev/null | head -1 || true)"
if [ -z "$LIB_DIR" ]; then
    echo "error: $SLICER_DIR has no lib/Slicer-<major.minor> directory" >&2
    exit 1
fi
SLICER_MINOR="$(basename "$LIB_DIR" | sed 's/^Slicer-//')"

# The revision only names the archive -- the install path is chosen by whichever
# Slicer opens it -- but matching Slicer's own naming makes the file self-
# describing when several are lying around.
REVISION="$(ls -d "$SLICER_DIR"/slicer.org/Extensions-* 2>/dev/null | head -1 |
            sed 's/.*Extensions-//' || true)"
[ -n "$REVISION" ] || REVISION="rev"

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
# Assemble.  The layout mirrors an installed extension exactly, which is what
# extractExtensionArchive copies into place.
# ---------------------------------------------------------------------------
ARCHIVE_BASE="${REVISION}-linux-amd64-${EXTENSION_NAME}-${VERSION}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/$ARCHIVE_BASE"
SCRIPTED="$ROOT/lib/Slicer-$SLICER_MINOR/qt-scripted-modules"
SHARE="$ROOT/share/Slicer-$SLICER_MINOR"
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
echo "  for Slicer  $SLICER_MINOR  (a package for one minor version is invisible to another)"
echo "  files       ${#SCRIPTS[@]} scripts + ${#RESOURCES[@]} resource(s)"
echo
echo "Install it: Slicer -> View -> Extensions Manager -> Install from file,"
echo "pick this archive, then restart.  GT Review appears under Segmentation."
