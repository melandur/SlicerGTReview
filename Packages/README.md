# Built packages

Extension archives that Slicer's Extensions Manager installs through
**View → Extensions Manager → Install from file**. Pick one, restart, and
*GT Review* appears under **Segmentation**. Nothing has to be built or compiled
to use them.

They are committed rather than ignored so a reviewer can install the tool from a
clone without a toolchain, and so it is always possible to see which build
someone is running.

To fetch one archive without cloning the repository, open it on GitHub and
press **Download raw file**, or use the raw form of its link, with the
archive's full name in place of `<file>`:

    curl -LO https://github.com/melandur/SlicerGTReview/raw/main/Packages/<file>

The ordinary link to the file (`…/blob/main/Packages/…`) returns GitHub's web
page about it, which Slicer cannot install.

## Reading the filename

    GTReview-for-Slicer-5.12-linux-amd64-v0.2.0-18-g3cffb8c.tar.gz
                        │    │           │
                        │    │           └─ git describe: tag, commits since, commit
                        │    └─ platform: linux, win or macosx
                        └─ the Slicer it installs on

Pick the file whose Slicer version matches yours (Help → About) and whose
platform matches your machine. Windows packages end in `.zip`, Linux and macOS
packages in `.tar.gz`.

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
What happens when an archive does not match depends on the platform:

- **Linux and Windows:** nothing fails loudly. Slicer copies whatever it finds,
  lists the extension as installed, and the module never appears. Uninstall it
  (Extensions Manager → Manage, and restart if it says "Uninstall pending
  restart") before installing the right one; a second install is otherwise
  refused as "already installed".
- **macOS:** the install fails with "Failed to copy directory" in the log,
  because the archive holds no `Extensions-<revision>/GTReview` tree for the
  running Slicer. Nothing is recorded as installed, so installing the right
  archive straight afterwards works.

Linux and Windows archives hold the same tree; only the container differs.
Windows packages are `.zip`, like every official Windows extension package, and
Linux packages are `.tar.gz`. Slicer reads both formats through libarchive, so
the platform tag in the name is descriptive only.

The exact commit is also written into the archive's `.s4ext` as `scmrevision`,
so an installed copy can be traced back even after the file is renamed. A
commit cannot say that the build held uncommitted changes, so the `git
describe` string of the file name, `-dirty` suffix included, goes into the same
file as a comment line:

    # gtreview-version v0.2.0-18-g3cffb8c

Slicer skips lines that start with `#` when it reads the file. GTReview reads
this one back and names the build in the header of every session in
`GTReview.log` as `v0.2.0-18-g3cffb8c (<scmrevision>)`. A package built before
the line existed shows the commit alone.

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

Linux and macOS archives need only `tar`. A Windows `.zip` is written with
Python 3's `zipfile` module — the script tries `python3`, `python`, then the
`PythonSlicer` of the `--slicer` installation — or, failing all of those, with
the `zip` command. Git Bash on Windows has neither `python3` nor `zip` by
default, so pass `--slicer` there. Either way the zip lists its top folder
first and every directory before its contents, the order Slicer relies on to
find the top-level directory.

Output lands here by default. The script refuses to build when the `.py` files
on disk and `MODULE_PYTHON_SCRIPTS` in `GTReview/CMakeLists.txt` disagree, in
either direction — an unlisted file keeps working from the source tree and
disappears only once packaged, which is a bad place to find out.
