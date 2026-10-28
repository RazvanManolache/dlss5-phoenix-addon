// Separate provider DLL: reuse the minimal host exports, not a GPU runtime.
#define wmain unused_load_test_main
#include "packaged_addon_load_tests.cpp"
#undef wmain
#if defined(_M_IX86)
#pragma comment(linker, "/EXPORT:ProviderBalanced=_ProviderBalanced")
#endif
extern "C" __declspec(dllexport) bool ProviderBalanced(bool attached) {
    return attached ? addons == 1 && events > 0 && overlays == 2
                    : addons == 0 && events == 0 && overlays == 0;
}
