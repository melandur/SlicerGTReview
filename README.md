# GTReview

A 3D Slicer extension for reviewing and correcting ground-truth segmentation masks
on brain-METS NIfTI datasets, **lesion by lesion**, without ever modifying the
original files.

Point it at a batch directory, step through the cases, inspect every connected
lesion in a sortable table, fix what is wrong with Slicer's own segment editor,
and write the result to a new `_reviewed_seg.nii.gz` beside the source data.

- **Category:** Segmentation
- **Module name:** GTReview
- **Requires:** 3D Slicer 5.10
- **License:** Apache License 2.0

---

## Guide for reviewers

`Docs/panel-guide.html` walks through every control in the panel, section by
section, in the order you meet them. Open it in a browser and keep it beside
Slicer; it is a single self-contained file, so it works from a file:// path or
from a share.

## Features

- **Dataset browser** — pick a batch directory, all cases are discovered
  automatically. Case combo box with `< Prev` / `Next >`, a `12 / 50` progress
  readout, a per-case reviewed checkmark, and a *skip already-reviewed* filter.
- **Multi-sequence display** — every image key found in a case (`t1`, `t1c`,
  `t2`, `flair`, `adc`, `dwi`, …) is loaded as its own volume. Choose which goes
  to background and which to foreground, set the foreground opacity, and switch
  layout (Sequences, Four-Up, 1x1 axial / sagittal / coronal, 2x2 slices,
  Conventional). *Sequences (axial)* shows every sequence of the case in its own
  axial view, ordered t1, t1c, t2, flair, then the rest; it is chosen
  automatically for cases with more than one image until you pick a layout.
- **Mask source picker** — start from `seg`, `gt`, `pred_seg`, any other mask
  present, an existing `reviewed_seg` (to resume), or an empty segmentation.
- **Contrast** — window/level for the image layer, with Slicer's auto/manual
  modes and modality presets. Display only; it never touches a voxel.
- **Slice alignment** — the slice views are rotated onto the mask's own voxel
  grid by default (`Slices: align to the image grid`). In this data the mask
  sits a median 8.3 degrees off the anatomical axes, so the default anatomical
  planes cut the voxel grid diagonally and a stroke drawn as a disc is
  committed as a stair-stepped one. Untick for true axial / coronal / sagittal.
- **Lesion list** — 26-connectivity connected components over the current mask,
  found after growing it by one voxel so fragments up to two voxels apart count
  as one lesion (voxel counts and volumes still cover only the real mask), in a
  sortable table of `#`, `Voxels`, `Volume (mm3)`, `Done` and a per-row
  delete button, default sorted by volume descending. Selecting a row jumps
  every slice view to that lesion's centre and highlights it. The centre is the
  centre of mass snapped to the nearest voxel that is actually *inside* the
  lesion, so it never lands in a hole or outside a crescent-shaped component.
  **Lesion numbers are a rank by size, not an identity**: they are reassigned on
  every recount, so a lesion's `#` changes when another lesion is painted,
  deleted or grows past it. *Done* is tracked by seed voxel, not by number, so
  the ticks survive the renumbering.
- **Editing tools** — `Active label` chooses what the brush lays down (label 1,
  2, 3, or Background, which is the Erase tool); `Paint over` chooses what
  may be overwritten (all labels, background only, or one named label) and
  drives the segment editor's masking. Then Paint, Erase and GTReview's own
  Sphere threshold (click the centre, drag to pull a sphere; tick **2D** to keep
  only the slice you drew on). "New lesion" paints a fresh component with the
  active label. `Brush: Live fill` (ticked) fills under the brush as you drag;
  untick it for Slicer's delayed paint, which draws outlines and fills on
  release and is far cheaper per stroke on large volumes. Plus Undo, Redo and
  Reset-to-loaded. Only labels 1 (Necrosis
  and Cavity), 2 (Enhancing Tumor) and 3 (Edema) exist and nothing in the UI
  can add another. Editing is locked until a lesion is
  selected (or New lesion is active).
- **Painting is immediate and undo is per stroke** — Slicer's delayed paint is
  off, so the segmentation follows the cursor instead of leaving outlined
  circles until the mouse comes up. Slicer saves an undo state per brush stamp
  rather than per stroke, so GTReview fingerprints the mask at mouse-down and
  one Undo press walks back to where the stroke began. Every other edit
  (lesion delete, label delete, a threshold apply) records a mark of its own, so
  one press is always one operation. There is a single undo stack: custom
  operations go through the segment editor's modify-selected-segment path so
  they land on the same stack as brush strokes.
- **Delete review** — next to the mask-source picker, enabled only when the case
  has a saved `reviewed_seg`. It removes that file from disk and reopens the
  case from its original mask; unlike everything else in the panel, `Ctrl+Z`
  does not reach it, so it names the file and asks first.
- **Save** — `Save & next case` at the bottom of the Editing section writes the reviewed mask and opens the next case; `Ctrl+S` saves without moving on. Both stay disabled until every lesion in the list is ticked *Done*.
- **Keyboard shortcuts** for the whole review loop (see below).

Pure logic — case discovery, connected components, NIfTI geometry-preserving
read/write — lives in `GTReviewLib/` with **no Slicer imports**, so it is unit
testable under plain `PythonSlicer`.

---

## Installation

The commands below use two variables so they can be pasted anywhere. Set them
once per shell:

```bash
export GTREVIEW=/path/to/this/repository      # the directory holding this README
export SLICER=/path/to/Slicer-5.10.0-linux-amd64
# macOS: SLICER=/Applications/Slicer.app/Contents  (the launcher is MacOS/Slicer)
# Windows: use Git Bash or WSL, SLICER=/c/Users/you/AppData/Local/NA-MIC/Slicer\ 5.10.0
```

### Developer path (no build — recommended for this repository)

You cannot build a Slicer extension against a downloaded Slicer *binary*: the
binary package ships no `SlicerConfig.cmake`, and `Slicer_DIR` must point at a
Slicer *build tree*. For day-to-day work you do not need to build anything —
scripted modules run straight from the source tree.

1. Launch Slicer.
2. **Edit → Application Settings → Modules → Additional module paths → Add**
3. Add the directory that **directly contains `GTReview.py`**:

   ```
   $GTREVIEW/GTReview
   ```

   Not the repository root. Module discovery is one level deep, not recursive —
   pointing at `$GTREVIEW` registers nothing and makes
   `GTReviewLib` unimportable.
4. Restart Slicer. `GTReview` now appears under **Segmentation** in the module
   list.

Equivalent one-shot invocation, no settings change:

```bash
$SLICER/Slicer \
    --additional-module-path $GTREVIEW/GTReview
```

Notes:

- The setting is stored in the **revision-specific** settings file
  (`slicer.app.revisionUserSettings()`, e.g.
  `<SlicerHome>/slicer.org/Slicer-34045.ini`, key `[Modules] AdditionalPaths`),
  **not** in `~/.config/slicer.org/Slicer.ini`. Editing the latter has no
  effect. Changing the setting requires a restart.
- When Slicer's scripted-module factory registers the module it appends the
  module's own directory to `sys.path`, which is what makes
  `import GTReviewLib` work — in both the source tree and an installed
  extension.
- **Reload caveat:** the *Reload* button re-executes `GTReview.py` only.
  Anything already imported from `GTReviewLib` stays cached in `sys.modules`.
  After editing `GTReviewLib/*.py`, restart Slicer (the module does not
  override `onReload`, so *Reload* will not pick those edits up).

### Install from a local package (no build tree)

`Packaging/make_package.sh` builds an archive the Extensions Manager accepts
through **Install from file**, straight from the source tree:

A prebuilt archive is already committed in [`Packages/`](Packages/) — install
that one and you need none of this. To build your own:

```bash
Packaging/make_package.sh --slicer $SLICER                 # this machine
Packaging/make_package.sh --slicer $SLICER --os win        # for a Windows user
Packaging/make_package.sh --slicer $SLICER --os macosx --revision 34045
```

Then in Slicer: **View → Extensions Manager → Install from file**, pick the
`.tar.gz`, restart. *GT Review* appears under **Segmentation**.

`--os` defaults to whatever the `--slicer` installation is, so the plain form
builds for the machine you are on. The layout differs per platform: macOS
buries the tree under `Slicer.app/Contents/Extensions-<revision>/<name>/`,
which is where `extractExtensionArchive` copies from there, while Linux and
Windows take it from the top level. That path embeds the **revision**, so a
macOS package is tied to one Slicer build rather than merely one minor version
— pass `--revision` matching the machine it is for, and the script refuses to
guess.

This works because a python-only extension needs no compilation: the manager
looks for a single top-level directory holding an `.s4ext` file (whose *name*
becomes the extension name) and copies the rest into place. The script writes
that description from the `EXTENSION_*` variables in the top-level
`CMakeLists.txt`, so it cannot drift from a real CPack build, and it refuses to
package if `MODULE_PYTHON_SCRIPTS` and the `.py` files on disk disagree in
either direction — the failure the module `CMakeLists.txt` warns about, where an
unlisted file keeps working from source and breaks only once installed.

The archive is tied to one Slicer **minor** version, because the module is
installed under `lib/Slicer-<major.minor>/qt-scripted-modules` and Slicer looks
nowhere else. Build one per target version; the directory name is read from the
`--slicer` installation rather than guessed.

### Extension path (built / installed package)

For end users the extension is installed like any other:

1. **View → Extensions Manager → Install Extensions**
2. Search for **GTReview**, install, restart Slicer.

To build a package yourself you need a Slicer **build tree** matching your
runtime, then:

```bash
cmake -DSlicer_DIR:PATH=/path/to/Slicer-build -S $GTREVIEW -B /path/to/GTReview-build
cmake --build /path/to/GTReview-build
cmake --build /path/to/GTReview-build --target package
```

The build must target the **same Slicer version** you intend to run: the module
installs under `lib/Slicer-<major.minor>/qt-scripted-modules/`, and a package
built against 5.11 is not picked up by a 5.10 runtime.

Installed layout (identical in shape to the source tree, which is why the icon
and resource lookups work in both):

```
<SlicerHome>/slicer.org/Extensions-<rev>/GTReview/lib/Slicer-5.10/qt-scripted-modules/
  GTReview.py
  GTReviewLib/{__init__,dataset,lesions,maskio}.py
  Resources/Icons/GTReview.png
```

> Every file under `GTReviewLib/` and every resource must be listed explicitly
> in `GTReview/CMakeLists.txt` — there is no globbing. A file you forget to list
> is *silently* omitted from the package: it works from the source tree and
> breaks only after installation.

---

## Expected data layout

```
<batch_dir>/
  batch_01_cases.txt              # free-text notes, ignored
  <case_id>/
    <case_id>_t1c.nii.gz          # image sequence
    <case_id>_seg.nii.gz          # ground-truth mask
    <case_id>_pred_seg.nii.gz     # model prediction mask
  <another_case_id>/
    ...
```

- Any sub-directory holding at least one `.nii` / `.nii.gz` becomes a case, and
  the **directory name is the case id**. Both `YG_78CQZ7VA3H2G_27` and
  `P39_2023-11-09` styles work; nothing about the id format is hard-coded.
- If the chosen root itself holds NIfTIs and contains no such sub-directory, the
  root is treated as a single case.
- For a file `<stem>.nii.gz` in a case directory the **key** is
  `stem[len(case_id) + 1:]` when the stem starts with `<case_id>_`, else the
  whole stem. Keys are classified case-insensitively, first match wins:

  | key | classified as |
  | --- | --- |
  | `reviewed_seg` | the review output — never offered as an input mask |
  | equals or ends with `seg`, `mask`, `label`, `labels`, `gt` | mask |
  | anything else | image sequence |

  So `pred_seg` is a mask, `t1c` is an image sequence.
- Any number of sequences may be present simultaneously or alone.
- Volumes in a case normally share geometry, but this is verified against the
  chosen reference rather than assumed.

---

## Output convention

Saving writes:

```
<case_dir>/<case_id>_reviewed_seg.nii.gz
```

- **Original files are opened read-only and are never overwritten.** The only
  file GTReview ever writes is `_reviewed_seg.nii.gz`.
- If `_reviewed_seg.nii.gz` already exists it is loaded as the starting mask, so
  a review can be resumed, and is overwritten on save after a confirmation
  prompt.
- The written mask preserves the source mask's **origin, spacing and direction
  exactly**, and uses an integer type — `uint8` unless the label values exceed
  255.
- A case counts as reviewed (checkmark in the browser, and the target of the
  *skip already-reviewed* filter) exactly when its `_reviewed_seg.nii.gz`
  exists.

---

## Keyboard shortcuts

Active while the GTReview module is the current module. They are suppressed
while a text field has focus, so typing in a line edit never triggers them.

| Shortcut | Action |
| --- | --- |
| `Ctrl+S` | Save the reviewed mask |
| `1` / `2` | Paint / erase (needs a selected lesion or an active "New lesion") |
| `3` | Sphere threshold: click a lesion's centre, drag to pull the sphere |
| `Esc` | Stop editing / cancel "New lesion" |
| `a` / `d` | Mask fill 10% more transparent / more opaque |
| `s` | Hide / show the mask |
| `Del` | Delete the selected lesion |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Redo |

The segment editor's built-in shortcuts are deliberately *not* installed: they
would bypass the lesion gate and their Ctrl+Z / Ctrl+Shift+Z duplicate ours (Qt
delivers an ambiguous shortcut to neither owner). For the same reason Slicer's
main-window Ctrl+S / Ctrl+Z / Ctrl+Y are muted while GTReview is the active
module and restored on exit.

---

## Running the tests

The pure-logic tests are plain `unittest` and need no Slicer application. They
import only the standard library and numpy — `slicer`, `vtk`, `qt` and `ctk`
exist solely inside a running Slicer, so anything `discover` picks up must stay
clear of them:

```bash
$SLICER/bin/PythonSlicer \
    -m unittest discover -s $GTREVIEW/Testing -v
```

(`pytest` is not installed in `PythonSlicer` by default; if you prefer it,
`PythonSlicer -m pip install pytest` first.)

The headless smoke test drives a real case end to end inside Slicer — load,
find lesions, delete one, save a `_reviewed_seg.nii.gz` to a temp directory,
re-read it:

```bash
$SLICER/Slicer --no-main-window \
    --python-script $GTREVIEW/Testing/smoke_headless.py
```

The integration test drives the module itself: it builds a synthetic case in a
temp directory, then loads it, paints with real mouse events on a slice view,
checks that one Undo press reverts the whole stroke, that Sphere threshold with
**2D** ticked stays inside the drawn slice, and that the per-row lesion delete
and *Delete review* behave. It needs the module on the path, and is named
without a `test_` prefix so the `discover` run above leaves it alone:

```bash
$SLICER/Slicer --no-splash \
    --additional-module-path $GTREVIEW/GTReview \
    --python-script $GTREVIEW/Testing/integration_gtreview.py
```

The in-application self test (`GTReviewTest`) is reachable from the module's
*Reload & Test* panel, and is registered as a ctest when the extension is built
with `BUILD_TESTING`.

---

## Publishing to the Extensions Index

Extensions are listed by submitting a **catalog entry** to
[`Slicer/ExtensionsIndex`](https://github.com/Slicer/ExtensionsIndex). The
short version:

1. **Host the source on GitHub.**
   - Repository name should be `Slicer` + extension name, i.e.
     **`SlicerGTReview`** — while the extension name itself must *not* start
     with `Slicer` (hence `EXTENSION_NAME GTReview`).
   - Add the GitHub topic **`3d-slicer-extension`**.
   - `LICENSE` in the repository root (Apache-2.0 here; MIT is the other
     recommended choice).
   - `README.md` in the root describing the extension, its modules, and how it
     works, with at least one screenshot.
   - Disable unused GitHub features (Wiki, Projects, Discussions, Packages).

2. **Make the top-level `CMakeLists.txt` correct.** It is load-bearing even
   though nothing is compiled: the Extensions Manager listing (description,
   homepage, icon, screenshots, contributors, status, category) is generated
   from its `EXTENSION_*` variables. Also confirm
   `EXTENSION_NAME` equals `PROJECT_NAME` — otherwise packaging is silently
   skipped — and that `EXTENSION_LICENSE_FILE` / `EXTENSION_README_FILE` point
   at this repository's files rather than defaulting to Slicer's own.

3. **Publish the icon and a screenshot** so `EXTENSION_ICONURL` and
   `EXTENSION_SCREENSHOTURLS` resolve. The icon is a **128x128 RGBA PNG**;
   `GTReview/Resources/Icons/GTReview.png` doubles as the module icon and the
   extension icon.

4. **Write the catalog entry** — a JSON file named after the extension,
   `GTReview.json`, at the root of a fork of `ExtensionsIndex`:

   ```json
   {
     "$schema": "https://raw.githubusercontent.com/Slicer/Slicer/main/Schemas/slicer-extension-catalog-entry-schema-v1.0.1.json#",
     "build_dependencies": [],
     "build_subdirectory": ".",
     "category": "Segmentation",
     "scm_revision": "main",
     "scm_type": "git",
     "scm_url": "https://github.com/melandur/SlicerGTReview",
     "tier": 1
   }
   ```

   Only `$schema`, `category` and `scm_url` are required. The schema is
   `"additionalProperties": false`, so `description`, `homepage`, `iconurl`,
   `screenshoturls`, `contributors` and `status` **must not** appear here —
   they live only in the top-level `CMakeLists.txt`. Prefer a branch name over a
   commit hash for `scm_revision`. New submissions use `"tier": 1`.

5. **Open the pull request against the right branch.** The `5.10` branch feeds
   the Slicer 5.10 stable release; `main` feeds Preview builds. Target `5.10`
   (and `main` if you also want Preview coverage). The `5.10` branch uses schema
   **v1.0.1**, which is the one that adds the `tier` field. Fill in the PR
   checklist that the template presents.

The legacy `.s4ext` description file is *not* hand-written or submitted — the
build generates one into the build tree from the `EXTENSION_*` variables, and
the `.s4ext` files found inside an installed extension are server-generated
install manifests.

---

## License

Copyright 2026 Neosoma Inc. Licensed under the Apache License, Version 2.0.
See [`LICENSE`](LICENSE).
