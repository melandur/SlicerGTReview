# Built packages

Extension archives that Slicer's Extensions Manager installs through
**View → Extensions Manager → Install from file**. Pick one, restart, and
*GT Review* appears under **Segmentation**. Nothing has to be built or compiled
to use them.

They are committed rather than ignored so a reviewer can install the tool from a
clone without a toolchain, and so it is always possible to see which build
someone is running.

## Reading the filename

    34627-linux-amd64-GTReview-v0.2.0-14-g64ae4e5.tar.gz
    │     │           │        │
    │     │           │        └─ git describe: tag, commits since, commit
    │     │           └─ extension name
    │     └─ target platform
    └─ Slicer revision(s); "34627+34645" means the archive serves both

Revisions map to releases: 34045 is 5.10.0, 34627 is 5.12.3, 34645 is 5.12.4.
Slicer shows its own under Help → About.

Two parts of that are load-bearing rather than descriptive:

- **A package works with one Slicer minor version only.** The module is
  installed under `lib/Slicer-<major.minor>/qt-scripted-modules` and Slicer
  looks nowhere else, so a 5.10 package is invisible to 5.12.
- **macOS packages are tied to Slicer *revisions*,** not merely a minor
  version, because the revision is part of the directory layout the manager
  copies from, and every patch release (5.12.3, 5.12.4, …) is a new revision.
  One archive can carry several revisions; `34627+34645` covers 5.12.3 and
  5.12.4. Linux and Windows archives are not tied to a revision, so a 34627
  Linux or Windows package also installs on 5.12.4.

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
    Packaging/make_package.sh --os linux --slicer-version 5.10 --revision 34045   # no install needed

Pass every revision a macOS archive should serve; the script writes one
`Extensions-<rev>/GTReview` tree per revision and Slicer picks its own.

Output lands here by default. The script refuses to build when the `.py` files
on disk and `MODULE_PYTHON_SCRIPTS` in `GTReview/CMakeLists.txt` disagree, in
either direction — an unlisted file keeps working from the source tree and
disappears only once packaged, which is a bad place to find out.
