#-----------------------------------------------------------------------------
set(MODULE_NAME GTReview)

#-----------------------------------------------------------------------------
# Every script and resource must be listed explicitly -- there is no globbing.
#
# slicerMacroBuildScriptedModule raises FATAL_ERROR at configure time for an
# entry that does not exist, but a file that exists and is NOT listed here is
# SILENTLY omitted from the packaged extension: it keeps working from the
# source tree and only breaks after packaging. Add every new GTReviewLib
# module to MODULE_PYTHON_SCRIPTS.
#
# The relative subdirectory of each entry is preserved verbatim when copied, so
# "GTReviewLib/dataset.py" lands next to "GTReview.py" as a real python package.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/dataset.py
  ${MODULE_NAME}Lib/layouts.py
  ${MODULE_NAME}Lib/lesions.py
  ${MODULE_NAME}Lib/maskio.py
  ${MODULE_NAME}Lib/SegmentEditorSphereThresholdEffect.py
  )

# Resources/ is shared across all scripted modules in the built tree, so the
# file names are namespaced by module name. ScriptedLoadableModule looks the
# icon up at "<dir of GTReview.py>/Resources/Icons/GTReview.png", which resolves
# both from the source tree and from the installed tree.
set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
# Do NOT pass WITH_SUBDIR: it nests the destination one directory deeper, which
# breaks the Resources/Icons lookup above and resourcePath().
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  # Registers the GTReviewTest ScriptedLoadableModuleTest subclass in
  # GTReview.py as a ctest. The pure-logic unit tests under <repo>/Testing are
  # run directly with PythonSlicer (see README) and are not part of the package.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
