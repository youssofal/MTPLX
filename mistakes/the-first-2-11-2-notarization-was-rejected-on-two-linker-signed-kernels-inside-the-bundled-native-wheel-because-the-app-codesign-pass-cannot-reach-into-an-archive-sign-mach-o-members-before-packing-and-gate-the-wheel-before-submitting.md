# The first 2.11.2 notarization was rejected on two linker-signed kernels inside the bundled native wheel, because the app codesign pass cannot reach into an archive: sign Mach-O members before packing and gate the wheel before submitting

**Symptom:** the release run passed pytest, the Swift suite, the pillar and
agent gates and built the signed app, then `notarytool submit --wait` came
back `status: Invalid` and `stapler` failed with error 65. The log
(`xcrun notarytool log <id> --keychain-profile mtplx-notary`) named two
files inside `Contents/Resources/Runtime/Native/mtplx-2.11.2-cp314-cp314-macosx_15_0_arm64.whl`:
`libmtplx_qsa_kernel_ops.dylib` and `_ext.cpython-314-darwin.so`, "not
signed with a valid Developer ID certificate" and "does not include a secure
timestamp". Both were ad-hoc linker-signed (`flags=0x20002(adhoc,linker-signed)`).

**Cause:** the notary service inspects every Mach-O it finds in the bundle,
archives included, while `codesign --deep` and the app build's explicit
signing walk only files on disk. The native sparse-prefill wheel (the #423
packaging) landed after 2.11.1, so this was the first time a wheel with
binaries went through notarization, and nothing signed its members.

**Fix / rule:** `scripts/bundle_native_runtime_wheel.py --codesign-identity`
signs each `.so`/`.dylib` member the way the app signs its binaries
(`--force --options runtime --timestamp --sign`), verifies the Developer ID
authority and the timestamp, and lets WheelFile rewrite RECORD for the signed
bytes. `scripts/release_macos_v1.sh` unpacks the bundled wheel and refuses to
continue unless every Mach-O carries both, so the failure costs seconds
instead of a 20-minute upload. Any new archive that carries binaries into
the bundle (wheels, zips, tarballs) needs the same two steps before the
first submission, not after.
