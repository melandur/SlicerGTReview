# Built packages

Extension archives that Slicer's Extensions Manager installs through
**View → Extensions Manager → Install from file**. Pick one, restart, and
*GT Review* appears under **Segmentation**. Nothing has to be built or compiled
to use them.

They are committed rather than ignored so a reviewer can install the tool from a
clone without a toolchain, and so it is always possible to see which build
someone is running.

## Reading the filename

    34045-linux-amd64-GTReview-v0.2.0-6-g7170fcb.tar.gz
    │     │           │        │
    │     │           │        └─ git describe: tag, commits since, commit
    │     │           └─ extension name
    │     └─ target platform
    └─ Slicer revision

Two parts of that are load-bearing rather than descriptive:

- **A package works with one Slicer minor version only.** The module is
  installed under `lib/Slicer-<major.minor>/qt-scripted-modules` and Slicer
  looks nowhere else, so a 5.10 package is invisible to 5.12.
- **macOS packages are tied to one Slicer *revision*,** not merely one minor
  version, because the revision is part of the directory layout the manager
  copies from. Linux and Windows are not.

The exact commit is also written into the archive's `.s4ext` as `scmrevision`,
so an installed copy can be traced back even after the file is renamed.

## Building one

    Packaging/make_package.sh --slicer $SLICER                 # this machine
    Packaging/make_package.sh --slicer $SLICER --os win
    Packaging/make_package.sh --slicer $SLICER --os macosx --revision 34045

Output lands here by default. The script refuses to build when the `.py` files
on disk and `MODULE_PYTHON_SCRIPTS` in `GTReview/CMakeLists.txt` disagree, in
either direction — an unlisted file keeps working from the source tree and
disappears only once packaged, which is a bad place to find out.
