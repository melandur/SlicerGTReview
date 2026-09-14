# Built packages

Extension archives that Slicer's Extensions Manager installs through
**View → Extensions Manager → Install from file**. Pick one, restart, and
*GT Review* appears under **Segmentation**. Nothing has to be built or compiled
to use them.

They are committed rather than ignored so a reviewer can install the tool from a
clone without a toolchain, and so it is always possible to see which build
someone is running.

## Reading the filename

    GTReview-for-Slicer-5.12-linux-amd64-v0.2.0-18-g3cffb8c.tar.gz
                        │    │           │
                        │    │           └─ git describe: tag, commits since, commit
                        │    └─ platform: linux, win or macosx
                        └─ the Slicer it installs on

Pick the file whose Slicer version matches yours (Help → About) and whose
platform matches your machine.

- **Linux and Windows names carry major.minor.** A `Slicer-5.12` package
  installs on 5.12.3, 5.12.4 and any later 5.12 release. It does not install
  on another minor version: the module lives under
  `lib/Slicer-<major.minor>/qt-scripted-modules` and Slicer looks nowhere else,
  so a 5.10 package is invisible to 5.12.
- **macOS names carry the exact releases,** for example
  `GTReview-for-Slicer-5.12.3-and-5.12.4-macosx-amd64-….tar.gz`. On macOS the
  release's revision is part of the directory layout the manager copies from,
  so the archive carries one tree per release it serves and installs on no
  other. Linux and Windows archives have no such tree.

Slicer never reads the file name; it is there for the person picking one.
Installing an archive that does not match does not fail loudly. Slicer copies
whatever it finds, lists the extension as installed, and the module never
appears. Uninstall it (Extensions Manager → Manage) before installing the right
one; a second install is otherwise refused as "already installed".

Linux and Windows archives are byte-for-byte the same layout; the platform tag
in the name is descriptive only. Windows accepts `.tar.gz` even though official
extensions ship as `.zip`.

The exact commit is also written into the archive's `.s4ext` as `scmrevision`,
so an installed copy can be traced back even after the file is renamed.

## Building one

    Packaging/make_package.sh --slicer $SLICER                      # this machine
    Packaging/make_package.sh --slicer $SLICER --os win
    Packaging/make_package.sh --slicer $SLICER --os macosx --revision 34627,34645
    Packaging/make_package.sh --os linux --slicer-version 5.10     # no install needed

A macOS archive needs every revision it should serve; the script writes one
`Extensions-<rev>/GTReview` tree per revision and Slicer picks its own.
Revisions map to releases: 34045 is 5.10.0, 34627 is 5.12.3, 34645 is 5.12.4.
The script names these releases in the file; a revision it does not know yet
appears as `5.12-r<rev>` until it is added to `release_for_revision` in
`Packaging/make_package.sh`. It refuses a revision whose release belongs to a
different minor version than the one being built.

Output lands here by default. The script refuses to build when the `.py` files
on disk and `MODULE_PYTHON_SCRIPTS` in `GTReview/CMakeLists.txt` disagree, in
either direction — an unlisted file keeps working from the source tree and
disappears only once packaged, which is a bad place to find out.
