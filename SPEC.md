# GTReview — 3D Slicer extension for segmentation review

## Purpose
Review and correct ground-truth segmentation masks for brain-METS NIfTI datasets,
lesion by lesion, without ever touching the original files.

## Environment (verified facts — do not re-derive)
- Slicer 5.10.0 at `$SLICER`
  - launcher `.../Slicer`, python `.../bin/PythonSlicer`
  - built-in scripted modules (READ THESE FOR REAL API USAGE):
    `$SLICER/lib/Slicer-5.10/qt-scripted-modules/`
  - Python 3.12.10, numpy, scipy 1.13.1, SimpleITK 2.5.2 all available in PythonSlicer.
- Project root: `$GTREVIEW` (git repo, currently empty except this SPEC).

## Data layout (verified)
Root example:
`<annotation-root>/METS/04_Groundtruthed/01_Yale/batch_01`

```
<batch_dir>/
  batch_01_cases.txt              # free-text notes, ignore
  <case_id>/
    <case_id>_t1c.nii.gz          # image sequence
    <case_id>_seg.nii.gz          # ground-truth mask (labels 0,1,2)
    <case_id>_pred_seg.nii.gz     # model prediction mask
```
- `case_id` == the sub-directory name. Two observed id styles:
  `YG_78CQZ7VA3H2G_27` and `P39_2023-11-09`. Never hard-code either.
- Suffix keys observed across the whole METS tree: `t1c`, `seg`, `pred_seg`.
  Must generically support any of `t1`, `t1c`, `t2`, `flair`, `adc`, `dwi`, ...
  present simultaneously or alone.
- All volumes in a case share geometry (shape 192x256x232, ~0.9mm iso in the sample),
  but the code must NOT assume that — resample/verify against the chosen reference.
- Mask label values observed: `{0, 1, 2}`. `pred_seg` may be all-zero.

## Naming rules (implement exactly)
For a file `<stem>.nii.gz` inside a case dir, `key = stem[len(case_id)+1:]` when the
stem starts with `case_id + "_"`, else `key = stem`.
Classify by key (case-insensitive), first match wins:
1. `reviewed_seg`  -> the review output; never listed as an input mask
2. key equals or ends with one of `seg`, `mask`, `label`, `labels`, `gt`  -> mask
3. otherwise -> image sequence

## Output contract
Saving writes `<case_dir>/<case_id>_reviewed_seg.nii.gz`.
- Original files are opened read-only and never overwritten.
- If `_reviewed_seg.nii.gz` already exists it is loaded as the starting mask
  (resume a review) and overwritten on save, after a confirmation prompt.
- Written mask must preserve the source mask's origin / spacing / direction exactly
  and use an integer type (uint8 unless labels exceed 255).

## Features
1. **Dataset browser** — pick a batch dir, discover all cases, combo box + Prev/Next,
   `12 / 50` progress, "skip already-reviewed" filter, per-case reviewed checkmark.
2. **Multi-sequence display** — every image key found becomes a loaded volume; user
   picks which goes to background/foreground; layout switcher (1x1, 2x2, 4-up).
3. **Lesion list** — 26-connectivity connected components over the current mask.
   Table columns: `#`, `Label`, `Voxels`, `Volume (mm3)`. Sortable, default by volume desc.
   Selecting a row jumps every slice view to that lesion's centre and highlights it.
   Centre = centre of mass snapped to the nearest voxel that is actually inside the lesion.
4. **Editing tools** — delete lesion, change a lesion's label, add segmentation by
   painting/erasing, threshold-based segmentation, undo, redo, reset-to-loaded.
5. **Save** — one button, "Save & next case", at the bottom of the Editing section (Ctrl+S saves without moving on). Both are enabled only once every lesion in the (up-to-date) list is ticked *Done*; an empty list counts as done.
6. **Keyboard shortcuts** — Ctrl+S save, Del delete selected
   lesion, Ctrl+Z undo, Ctrl+Y / Ctrl+Shift+Z redo,
   1 / 2 paint / erase, 3 sphere threshold, Esc stop editing,
   a / d mask 10% more transparent / opaque, s hide+show mask.
7. **Editing is gated** — the brush, Delete and the threshold effects only work
   while a lesion is selected in the table, or while "New lesion → Start painting"
   is active.  Only labels 1 (Necrosis and Cavity), 2 (Enhancing Tumor) and
   3 (Edema) exist; all three segments are always present so any can be
   painted, and nothing in the UI can add a fourth.  **Sphere threshold** (a GTReview Segment Editor effect):
   click the centre of a lesion — that voxel's intensity ± a tolerance is the
   range — drag to pull a physical sphere, release to add every in-range voxel in
   the sphere that is connected to the seed; one undo step, plain voxel mask.  The
   3D surface is unsmoothed so it follows the voxels.

## Architecture decisions (binding — do not re-litigate)
- **The mask lives in a `vtkMRMLSegmentationNode`**, one segment per label value.
  All editing goes through an embedded `qMRMLSegmentEditorWidget`, which supplies
  paint / erase / sphere threshold AND a single coherent undo-redo stack.
  Do NOT hand-roll a second undo system; wire the Undo/Redo buttons to the editor's.
- Custom operations (delete lesion, relabel lesion) must be applied through the
  segment editor's modify-selected-segment path so they land on the same undo stack.
- GUI is built **programmatically in Python** with `qt`/`ctk`/`slicer` widgets.
  No `.ui` file, no Qt Designer round-trip.
- Pure logic (case discovery, connected components) lives in importable modules with
  no Slicer imports, so it is unit-testable under plain PythonSlicer.

## File layout (binding)
```
gt_tool_slicer/
  SPEC.md
  README.md
  CMakeLists.txt                       # extension-level
  GTReview/
    CMakeLists.txt                     # module-level
    GTReview.py                        # ScriptedLoadableModule: GTReview, GTReviewWidget, GTReviewLogic, GTReviewTest
    Resources/Icons/GTReview.png
    GTReviewLib/
      __init__.py
      dataset.py                       # case discovery  (NO slicer imports)
      lesions.py                       # connected components (NO slicer imports)
      maskio.py                        # nifti read/write preserving geometry (SimpleITK only)
  Testing/
    test_dataset.py
    test_lesions.py
    test_maskio.py
    smoke_headless.py                  # runs inside Slicer --no-main-window
```

## Python API contract (binding — implement these signatures verbatim)

### GTReviewLib/dataset.py
```python
IMAGE, MASK, REVIEWED = "image", "mask", "reviewed"

@dataclass
class Case:
    case_id: str
    directory: str
    images: dict[str, str]      # key -> abs path, e.g. {"t1c": "/.../x_t1c.nii.gz"}
    masks: dict[str, str]       # key -> abs path, e.g. {"seg": ..., "pred_seg": ...}
    reviewed_path: str          # <directory>/<case_id>_reviewed_seg.nii.gz (may not exist)

    @property
    def is_reviewed(self) -> bool: ...
    def default_mask_path(self, preferred=("seg", "gt", "pred_seg")) -> str | None:
        """reviewed_path if it exists, else first preferred key present, else any mask."""

def classify_key(key: str) -> str: ...            # -> IMAGE | MASK | REVIEWED
def parse_case_files(case_dir: str, case_id: str | None = None) -> Case: ...
def discover_cases(root: str) -> list[Case]:
    """Sub-dirs containing >=1 .nii/.nii.gz become cases, sorted by case_id.
    If root itself holds niftis and no such sub-dir exists, root is a single case."""
```

### GTReviewLib/lesions.py
```python
@dataclass
class Lesion:
    index: int                 # 1-based, stable for a given components run
    label: int                 # dominant mask label value inside the component
    voxel_count: int
    volume_mm3: float
    centroid_ijk: tuple[int, int, int]   # i,j,k voxel INSIDE the lesion
    bbox_ijk: tuple[tuple[int,int], tuple[int,int], tuple[int,int]]

def find_lesions(mask_ijk, spacing_ijk, connectivity=26, min_voxels=1) -> tuple[np.ndarray, list[Lesion]]:
    """mask_ijk: 3-D int array indexed [i,j,k]. Returns (component_label_map, lesions)
    sorted by voxel_count descending. Component map is 0 for background, n for lesion n."""

def lesion_mask(component_map, index) -> np.ndarray: ...
```
Note: components are found over the **binarised** mask (any non-zero voxel), so a
lesion spanning two label values stays one lesion; `Lesion.label` reports the
most frequent non-zero value inside it.

### GTReviewLib/maskio.py
```python
@dataclass
class MaskGeometry: origin; spacing; direction; size   # SimpleITK conventions

def read_mask(path) -> tuple[np.ndarray, MaskGeometry]:   # array indexed [i,j,k]
def write_mask(path, array_ijk, geometry, dtype=np.uint8) -> None
```
Array index order is **[i,j,k]** everywhere in this codebase (SimpleITK's
`GetArrayFromImage` returns [k,j,i], so transpose at the boundary — do it in maskio
and nowhere else).

## Definition of done
- `PythonSlicer -m pytest Testing/` passes (or the plain-unittest equivalent).
- `Slicer --no-main-window --python-script Testing/smoke_headless.py` loads a real case
  from batch_01, finds lesions, deletes one, saves a `_reviewed_seg.nii.gz` to a temp
  dir, and re-reads it — exits 0.
- The module appears under a `Segmentation` category in Slicer's module list.
