#include "phoenix_native.h"
int main(void) {
    PhoenixNativeControls controls;
    phoenix_native_default_controls(&controls);
    return controls.struct_bytes == sizeof(controls) && controls.local_tone == 1 ? 0 : 1;
}
