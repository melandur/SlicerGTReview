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
#     machine it is for.  Each 5.x.y patch release is a new revision (5.12.3 is
#     34627, 5.12.4 is 34645), so list every revision the archive should serve.
#
# Windows packages are .zip and the others .tar.gz, the formats the extensions
# server ships for each platform.  Slicer reads either on any platform through
# libarchive (vtkArchive::ExtractTar enables every format), so the container
# follows the official packages rather than what installs.  What does matter is
# the order of the entries: extractArchive() takes the top-level directory from
# the FIRST entry the archive lists, walking up from that entry's folder, so the
# top folder is written first and every directory before its contents.
#
# An archive with the wrong layout does not always fail loudly.  On Linux and
# Windows Slicer copies whatever it finds, records the extension as installed,
# and the module simply never appears; a second attempt with the right archive
# is then refused as "already installed" until the first one is uninstalled.
# On macOS the copy from Extensions-<revision>/<name> fails instead ("Failed to
# copy directory"), nothing is recorded, and a retry with the right archive
# works.
#
# Usage:
#   Packaging/make_package.sh [--slicer <dir>] [--os linux|macosx|win]
#                             [--revision <rev>] [--slicer-version <x.y>]
#                             [--output <dir>]
#
#   --slicer          a Slicer installation to read the layout from
#   --os              target platform (default: detected from --slicer, else this host)
#   --revision        target Slicer revision(s), e.g. 34627 or 34627,34645.
#                     Cosmetic on Linux and Windows, LOAD-BEARING on macOS (see
#                     above).  A comma-separated list makes ONE macOS archive
#                     that installs on any of the listed revisions: the manager
#                     copies only Extensions-<running revision>/<name> and
#                     ignores the rest
#   --slicer-version  major.minor, e.g. 5.10, when building without a --slicer
#   --output          where to write the archive (default: <repo>/Packages)
#
# Output: <output>/GTReview-for-Slicer-<version>-<os>-amd64-<git describe>.<ext>
#   <version> is major.minor on Linux and Windows (the package serves every
#   patch release of it) and the exact releases on macOS, e.g. 5.12.3-and-5.12.4.
#   <ext> is zip on Windows and tar.gz on Linux and macOS.  A zip needs Python 3
#   (python3, python, or the --slicer installation's PythonSlicer) or the zip
#   command.  Slicer never reads the file name, so it is there for the person
#   picking one.
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLICER_DIR="${SLICER_HOME:-}"
OUTPUT_DIR="$REPO/Packages"
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
        # Print the whole header comment, however long it grows.
        -h|--help) sed -n '2,/^[^#]/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"; exit 0 ;;
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
IFS=, read -r -a REVISIONS <<< "$REVISION"

# Slicer release of a revision, for the file name.  Unknown revisions keep
# their number (5.12-r34700) rather than guessing a release.
release_for_revision() {
    case "$1" in
        34045) echo 5.10.0 ;;
        34627) echo 5.12.3 ;;
        34645) echo 5.12.4 ;;
        *) echo "${SLICER_MINOR}-r$1" ;;
    esac
}
if [ "$TARGET_OS" = macosx ]; then
    RELEASES=()
    for rev in "${REVISIONS[@]}"; do
        release="$(release_for_revision "$rev")"
        case "$release" in
            "$SLICER_MINOR".*|"$SLICER_MINOR"-r*) ;;
            *) echo "error: revision $rev is Slicer $release, not $SLICER_MINOR" >&2; exit 2 ;;
        esac
        RELEASES+=("$release")
    done
    SLICER_LABEL="$(printf '%s-and-' "${RELEASES[@]}")"
    SLICER_LABEL="${SLICER_LABEL%-and-}"
else
    SLICER_LABEL="$SLICER_MINOR"
fi
ARCHIVE_BASE="${EXTENSION_NAME}-for-Slicer-${SLICER_LABEL}-${TARGET_OS}-amd64-${VERSION}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/$ARCHIVE_BASE"
if [ "$TARGET_OS" = macosx ]; then
    INNERS=()
    for rev in "${REVISIONS[@]}"; do
        INNERS+=("$ROOT/Slicer.app/Contents/Extensions-$rev/$EXTENSION_NAME")
    done
else
    INNERS=("$ROOT")
fi

populate() {
    local inner="$1"
    local scripted="$inner/lib/Slicer-$SLICER_MINOR/qt-scripted-modules"
    local share="$inner/share/Slicer-$SLICER_MINOR"
    mkdir -p "$scripted" "$share"
    for script in "${SCRIPTS[@]}"; do
        mkdir -p "$scripted/$(dirname "$script")"
        cp "$MODULE_DIR/$script" "$scripted/$script"
    done
    for resource in "${RESOURCES[@]}"; do
        mkdir -p "$scripted/$(dirname "$resource")"
        cp "$MODULE_DIR/$resource" "$scripted/$resource"
    done
    write_s4ext "$share/$EXTENSION_NAME.s4ext"
}

# Key-value pairs, first token is the key, '#' starts a comment.  Parsed by
# qSlicerExtensionsManagerModel::parseExtensionDescriptionFile.
#
# scmrevision is the commit, which cannot tell that the tree held uncommitted
# changes when the package was built.  The git describe string does (a -dirty
# suffix), so it rides along in a comment line: Slicer skips it, and GTReview
# reads it back for the header of GTReview.log (sessionlog.installed_build).
# The parser skips a line only when '#' is its very first character (5.12.3
# checks line.startsWith('#') before splitting), so the line must not be
# indented.
write_s4ext() {
    cat > "$1" <<EOF
# Generated by Packaging/make_package.sh from the top-level CMakeLists.txt.
# The extension name is taken from THIS FILE'S NAME, not from any key here.
# gtreview-version $VERSION
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
}

for inner in "${INNERS[@]}"; do
    populate "$inner"
done

# ---------------------------------------------------------------------------
# Pack.  A Windows zip needs a tool that exists on Linux, macOS and Git Bash on
# Windows alike.  Python's zipfile module comes first: it is on nearly every
# Linux and macOS machine, and every Slicer ships a Python, which matters in Git
# Bash where neither python3 nor zip is installed by default.  Each candidate is
# run rather than trusted from `command -v`, because Windows puts a python3.exe
# on PATH that only opens the Microsoft Store, and an old macOS python is
# Python 2, which lacks ZipInfo.from_file.
# ---------------------------------------------------------------------------
find_zip_python() {
    local candidate
    local candidates=(python3 python)
    if [ -n "$SLICER_DIR" ]; then
        candidates+=("$INSTALL_PREFIX/bin/PythonSlicer" "$INSTALL_PREFIX/bin/PythonSlicer.exe")
    fi
    for candidate in "${candidates[@]}"; do
        if "$candidate" -c 'import zipfile, zlib; zipfile.ZipInfo.from_file' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# Depth first in name order, each directory's entry before anything inside it
# and the top folder first, because extractArchive() reads the top-level
# directory off the first entry.  zipfile writes the forward slashes the format
# requires whatever the host's separator, and marks directory entries as such.
ZIP_PY='import os, sys, zipfile
top = sys.argv[1]
out = sys.argv[2]

def add(archive, path):
    archive.write(path, path)
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            add(archive, os.path.join(path, name))

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
    add(archive, top)
'

# The zip is written inside the stage and moved into place afterwards, so both
# tools only ever see relative names: a native Windows python would not
# understand a /c/Users/... path.  The MSYS variables stop Git Bash (and MSYS2)
# from rewriting arguments that look like POSIX paths on their way to it.
make_zip() {
    local python
    if python="$(find_zip_python)"; then
        (cd "$STAGE" && MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' \
            "$python" -c "$ZIP_PY" "$ARCHIVE_BASE" "$ARCHIVE_BASE.zip")
    elif command -v zip >/dev/null 2>&1; then
        # A path sorts after every path that is a prefix of it, so a plain sort
        # lists the top folder first and each directory before its contents.
        (cd "$STAGE" && find "$ARCHIVE_BASE" | LC_ALL=C sort | zip -q -X "$ARCHIVE_BASE.zip" -@)
    else
        echo "error: a Windows package is a .zip, and building one needs Python 3" >&2
        echo "       (python3 or python, with its zipfile module) or the zip command;" >&2
        echo "       none was found.  Install one, or pass --slicer <installation>" >&2
        echo "       so that its PythonSlicer is used." >&2
        exit 1
    fi
}

mkdir -p "$OUTPUT_DIR"
if [ "$TARGET_OS" = win ]; then
    ARCHIVE="$OUTPUT_DIR/$ARCHIVE_BASE.zip"
    make_zip
    mv -f "$STAGE/$ARCHIVE_BASE.zip" "$ARCHIVE"
else
    ARCHIVE="$OUTPUT_DIR/$ARCHIVE_BASE.tar.gz"
    tar -czf "$ARCHIVE" -C "$STAGE" "$ARCHIVE_BASE"
fi

echo "built $ARCHIVE"
echo
echo "  extension   $EXTENSION_NAME $VERSION"
if [ "$TARGET_OS" = macosx ]; then
    echo "  target      macosx, Slicer ${RELEASES[*]} (revision ${REVISIONS[*]})"
else
    echo "  target      $TARGET_OS, Slicer $SLICER_MINOR (any $SLICER_MINOR.x)"
fi
echo "  files       ${#SCRIPTS[@]} scripts + ${#RESOURCES[@]} resource(s)"
if [ "$TARGET_OS" = macosx ]; then
    echo "  note        macOS packages are tied to the revision(s) $REVISION, not just to $SLICER_MINOR"
else
    echo "  note        a package for Slicer $SLICER_MINOR is invisible to any other minor version"
fi
echo
echo "Install it: Slicer -> View -> Extensions Manager -> Install from file,"
echo "pick this archive, then restart.  GT Review appears under Segmentation."
