# 0.10 validation scope

- The compatibility changes were built from maintained source for x86/x64;
  the build ran 21 CPU contract suites and two software-D3D12 suites.
- All 82 Windows synthetic fixtures passed installation/rendering checks during
  development, including ten software-device cases. Success required processed,
  presented frames and a measurable output difference, not just installed files.
- Those runs used successive builds; they are not a final-build fresh-install sweep.
- The 0.9 neural kernels and standalone portable libraries are retained unchanged.
  Their earlier validation and timing records remain historical evidence.
- Release assembly recursively audits embedded payloads and source files, checks
  archive integrity and excludes model data, personal paths and configuration.

No fresh full game-library sweep, retail-game qualification, benchmark rerun,
AMD/Intel hardware qualification or physical-phone test is claimed by this update.
DOS dependency provisioning belongs to Everything and remains separately qualified.
