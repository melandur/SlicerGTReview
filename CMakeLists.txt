cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

#-----------------------------------------------------------------------------
# EXTENSION_NAME must equal PROJECT_NAME, otherwise SlicerExtensionCPack.cmake
# silently prints "Skipping extension packaging" and returns without an error.
# The name must also not be empty, "Slicer" or "Project" (UseSlicer.cmake.in
# raises FATAL_ERROR for those), and by convention it does NOT start with
# "Slicer" even though the GitHub repository does (SlicerGTReview).
set(EXTENSION_NAME GTReview)
project(${EXTENSION_NAME})

#-----------------------------------------------------------------------------
# Extension meta-information.
#
# NOTE: these variables are load-bearing even though nothing is compiled for a
# python-only extension: the Extensions Manager listing is generated from them
# (via the .s4ext description file). The Extensions Index catalog entry
# (GTReview.json) has "additionalProperties": false and cannot carry them.
#
set(EXTENSION_HOMEPAGE "https://github.com/melandur/SlicerGTReview")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Neosoma Inc. (https://www.neosomainc.com)")
set(EXTENSION_DESCRIPTION "Review and correct ground-truth segmentation masks for brain-METS NIfTI datasets, lesion by lesion, without ever modifying the original files.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/melandur/SlicerGTReview/main/GTReview/Resources/Icons/GTReview.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/melandur/SlicerGTReview/main/Screenshots/1.png")
set(EXTENSION_STATUS "Beta")
set(EXTENSION_DEPENDS "NA")            # space separated list of extension names, or "NA"
set(EXTENSION_BUILD_SUBDIRECTORY ".")  # python-only: no superbuild, no inner-build
set(EXTENSION_ENABLED 1)

# Ship OUR license and readme. Without these two, UseSlicer.cmake.in silently
# falls back to Slicer's own LICENSE/README and CPack would present Slicer's
# license as this extension's EULA.
set(EXTENSION_LICENSE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(EXTENSION_README_FILE  "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(GTReview)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
