#!/bin/bash
#
# Let one Slicer.app use the microphone on macOS, for GTReview's recordings.
#
# macOS hands an app the microphone only if its Info.plist says what it is
# for (NSMicrophoneUsageDescription) and -- the official Slicer builds are
# signed with the hardened runtime -- if its signature carries the
# com.apple.security.device.audio-input entitlement.  Slicer has neither, so
# macOS gives it silence, without asking.  This script adds both to the
# Slicer.app it is given and signs that app again, on this Mac (ad hoc),
# keeping the entitlements Slicer had.  macOS then asks once, the first time
# GTReview opens the microphone.
#
# Quit Slicer first, then:
#
#     bash GTReviewMacMicrophone.sh /Applications/Slicer.app
#
# Run it again after installing a new Slicer.  Installing Slicer again undoes
# it.  An app the current user cannot write needs sudo in front.
set -euo pipefail

APP="${1:-}"
APP="${APP%/}"
if [[ -z "$APP" || ! -f "$APP/Contents/Info.plist" || ! -d "$APP/Contents/MacOS" ]]; then
    echo "usage: bash $0 /path/to/Slicer.app" >&2
    exit 2
fi
if [[ "$(uname)" != "Darwin" ]]; then
    echo "error: this is for macOS" >&2
    exit 2
fi
# (not just anything under the app: this script itself may live in its
# Extensions folder)
if pgrep -f "$APP/Contents/bin/SlicerApp-real" > /dev/null 2>&1; then
    echo "error: Slicer is running from $APP -- quit it first" >&2
    exit 1
fi
PLIST="$APP/Contents/Info.plist"
if [[ ! -w "$PLIST" ]]; then
    echo "error: $APP cannot be changed by $(whoami); run this again with sudo in front" >&2
    exit 1
fi

WHY="GTReview records your voice during a review, so that whoever listens later knows which lesion you meant."
WORK="$(mktemp -d)"
cp -p "$PLIST" "$WORK/Info.plist.original"
cleanup() {
    status=$?
    if [[ $status -ne 0 && -f "$WORK/Info.plist.original" ]]; then
        # signing failed half way: put Slicer's own Info.plist back, which its
        # signature still matches
        cp -p "$WORK/Info.plist.original" "$PLIST"
        echo "error: nothing was changed that Slicer's own signature does not cover" >&2
    fi
    rm -rf "$WORK"
}
trap cleanup EXIT

# Sign $1 again, ad hoc and with the hardened runtime, with the entitlements
# it has plus the microphone.
sign() {
    local target="$1" entitlements="$WORK/entitlements.plist"
    codesign -d --entitlements :- "$target" > "$entitlements" 2> /dev/null || true
    if ! /usr/libexec/PlistBuddy -c "Print" "$entitlements" > /dev/null 2>&1; then
        rm -f "$entitlements"  # none: start an empty one
        /usr/libexec/PlistBuddy -c "Clear dict" "$entitlements" > /dev/null
    fi
    /usr/libexec/PlistBuddy -c "Delete :com.apple.security.device.audio-input" "$entitlements" > /dev/null 2>&1 || true
    /usr/libexec/PlistBuddy -c "Add :com.apple.security.device.audio-input bool true" "$entitlements"
    codesign --force --sign - --options runtime --entitlements "$entitlements" "$target"
}

echo "Slicer:      $APP"
# SlicerApp-real is the process that opens the microphone ...
if [[ -f "$APP/Contents/bin/SlicerApp-real" ]]; then
    sign "$APP/Contents/bin/SlicerApp-real"
    echo "signed:      Contents/bin/SlicerApp-real"
fi
# ... the launcher in Contents/MacOS is the app macOS asks for, whose
# Info.plist must say why, and whose signature seals that Info.plist
/usr/libexec/PlistBuddy -c "Delete :NSMicrophoneUsageDescription" "$PLIST" > /dev/null 2>&1 || true
/usr/libexec/PlistBuddy -c "Add :NSMicrophoneUsageDescription string '$WHY'" "$PLIST"
sign "$APP"
echo "signed:      the app, with NSMicrophoneUsageDescription in its Info.plist"
# an app downloaded with a browser would otherwise be checked again, as unsigned
xattr -dr com.apple.quarantine "$APP" 2> /dev/null || true

codesign -d --entitlements :- "$APP" 2> /dev/null | grep -q "com.apple.security.device.audio-input" ||
    { echo "error: the entitlement is not in the new signature" >&2; exit 1; }
rm -f "$WORK/Info.plist.original"
echo "Done.  Start Slicer: macOS asks once whether Slicer may use the microphone."
