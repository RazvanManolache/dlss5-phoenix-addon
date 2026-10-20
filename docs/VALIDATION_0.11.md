# 0.11 validation scope

- Fresh source compilation and packaging cover both x86 and x64 addons, 22 CPU
  contract suites and two software-D3D12 suites.
- CPU routing checks cover both architectures, available choices, configured
  absolute/relative folders, Unicode paths, DLL architecture, package/profile
  hashes, native-driver graph readiness and declared model support.
- Five targeted generated DX12 fixture cases check x86 automatic library
  selection, explicit x86 native driver, automatic portable selection from an
  alternate folder, final helper fallback and x64 same-device native execution.
  Acceptance requires successful inference, a fresh presented revision, output
  progress and independently changed scene pixels. Owned fixtures are closed.
- These reset-frame fixture checks do not establish temporal correctness,
  cross-route output parity or retail-game compatibility. Experimental native
  x86 profiles used for testing remain external to the release.
- Source archives and recursively embedded binary payloads are audited for
  excluded models, private paths, configuration and logs; archive hashes and
  public downloads are verified. The attached validation receipt identifies
  the exact source commit, binaries and checks.

The preceding broader 82-fixture development run used successive older builds.
It is not relabeled as a fresh 0.11 full-matrix result. Further installation
testing is separate and continues after publication. No new inference benchmark,
AMD/Intel hardware, headset or physical-phone qualification is claimed.
