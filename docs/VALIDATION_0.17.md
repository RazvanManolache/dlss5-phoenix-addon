# Phoenix DLSS5 0.17 validation

## Fix qualification

- Startup-refusal recovery: 443 route-policy checks on each of x86 and x64. Enabled fallback, disabled fallback, discovery waits, protected resources, queue ownership and alternate-route ordering are covered.
- Three WARP D3D11-on-12 x86 cases retain their explicit library, driver and portable preferences with fallback enabled. All passed using an independently identified x64 helper, with current presented output, central-pixel changes, continuing frames and clean shutdown. The tested standard x86 binary is bound in the release receipt. Strict fallback-disabled baseline refusals are preserved.
- Phoenix Input: actual 20-cycle reload probes on both architectures showed no heap or handle growth after warm-up with the repaired binaries. Six actual file-lock/malformed-batch checks passed; six paired-capture/model-switch cases plus four focused Phoenix/NVIDIA capture checks passed. These bounded observations do not prove every long-running workload leak-free.
- Experimental x86 driver: seven targeted repaired-runtime checks passed. Four API comparisons were byte-identical to their accepted baseline outputs. The source change caps temporary weight uploads at 4 MiB; full device weights are unchanged. The tested Windows/driver-specific WOW64 supplement is not included in the public packages.

## Final package gates

The standard and experimental packages bind their final x86/x64 addons and helper to their build receipts. Each build runs 24 CPU contract suites and three software D3D12 suites. Final architecture-matched minimal ReShade hosts check real DLL registration, public ABI, balanced detach and unloading. Embedded resources are recursively audited and hash checked, including the nested x64 addon and graphics helper.

The source archive equals the audited Git tree. Source, binary payloads and nested resources are scanned for excluded models, NVIDIA DLLs, private host paths, credentials and diagnostic artifacts. Optional Input binaries retain their matching repaired source/build and live-test receipts. Portable inference companions and native kernels retain their previous identities; they are not represented as newly benchmarked.

`VALIDATION.json` binds the publication commit, final downloadable binary hashes, package gates and applicable targeted evidence. Public downloads are independently retrieved and compared with the prepared SHA-256 checksums before publication is considered complete.

## Boundaries

The preceding full acceptance run exercised the released 0.16 main addons, supplemental development runtimes and repaired Input/driver candidates separately. Its 849 entries and 6.3 hours of memory observation remain evidence for those exact tested inputs. No full 0.17 graphics matrix, new retail-game check, physical older-RTX qualification or fresh performance benchmark is claimed for this publication. Fallback recovery on the rebuilt standard package has the three targeted live checks above; experimental-package release gates are CPU/software checks.
