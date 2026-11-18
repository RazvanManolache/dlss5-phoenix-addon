# Phoenix 1.3.3 validation

- Shared C++/ImGui controls passed on x86 and x64 at 360, 700, and 1100 pixel widths. Checks cover merged GPU identification, shared-device deduplication, failed-route messaging, and disabled processing.
- Windows builds check 35 CPU suites and seven software D3D12 suites, followed by final x86/x64 addon loading, resource extraction, and embedded helper verification.
- Live DX11 fixture checks exercised AMD game rendering with RTX 5080 helper inference, RTX 5080 in-process processing, and RTX 5080 helper processing. Pausing and resuming the owned helper verified that its inference wait appeared and cleared after output resumed.
- Gameplay captures exercised Quake II Enhanced through in-process D3D12 and the original Quake III executable through the x64 helper. The ioquake3 launch attempts failed. Final status-box consolidation was checked with the shared UI tests.
- Release packages retain checksum-verified model-free inference payloads from 1.3.2. Linux release CI builds and tests the Linux layer and providers before publication.
