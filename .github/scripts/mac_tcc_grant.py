"""Record in macOS's privacy database that an app may use the microphone (CI only).

What a click on "Allow" in macOS's microphone prompt does, for a machine
nobody sits at.  Writes the user's and the system's TCC.db where it may; how
far that gets depends on the machine (System Integrity Protection, Full Disk
Access).  Prints what happened; never fails the job by itself.

    python3 mac_tcc_grant.py /path/to/Slicer.app
"""
import os
import plistlib
import sqlite3
import subprocess
import sys
import tempfile
import time

app = sys.argv[1]
with open(os.path.join(app, "Contents", "Info.plist"), "rb") as handle:
    bundle_id = plistlib.load(handle)["CFBundleIdentifier"]
print("app:", app, "bundle id:", bundle_id)

# the app's designated requirement, compiled, as macOS stores it with a
# decision, so the decision holds for exactly this signature
csreq = None
shown = subprocess.run(["codesign", "-d", "-r-", app], capture_output=True, text=True)
requirement = next((line.split("=>", 1)[1].strip() for line in (shown.stdout + shown.stderr).splitlines()
                    if "designated =>" in line), None)
print("requirement:", requirement)
if requirement:
    with tempfile.TemporaryDirectory() as work:
        blob = os.path.join(work, "csreq.bin")
        if subprocess.run(["csreq", "-r-", "-b", blob], input=requirement, text=True).returncode == 0:
            with open(blob, "rb") as handle:
                csreq = handle.read()

for database in (os.path.expanduser("~/Library/Application Support/com.apple.TCC/TCC.db"),
                 "/Library/Application Support/com.apple.TCC/TCC.db"):
    try:
        connection = sqlite3.connect(database)
        columns = [row[1] for row in connection.execute("PRAGMA table_info(access)")]
        values = {
            "service": "kTCCServiceMicrophone", "client": bundle_id, "client_type": 0,
            "auth_value": 2, "auth_reason": 3, "auth_version": 1, "csreq": csreq, "policy_id": None,
            "indirect_object_identifier_type": 0, "indirect_object_identifier": "UNUSED",
            "indirect_object_code_identity": None, "flags": 0, "last_modified": int(time.time()),
            "pid": None, "pid_version": None, "boot_uuid": "UNUSED", "last_reminded": 0,
        }
        connection.execute(
            "INSERT OR REPLACE INTO access ({}) VALUES ({})".format(
                ",".join(columns), ",".join("?" * len(columns))),
            [values.get(column) for column in columns])
        connection.commit()
        connection.close()
        print("granted in", database)
    except Exception as error:  # noqa: BLE001 - reported, the next database may work
        print("could not write {}: {}".format(database, error))
