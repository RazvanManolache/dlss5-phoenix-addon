// Compile as C, link the published ABI, and verify failures without a model.
#include "phoenix.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    if (phoenix_runtime_create(NULL, 64, 64) != NULL ||
        strstr(phoenix_runtime_error(), "configuration") == NULL)
        return 1;
    if (phoenix_runtime_create("{broken", 64, 64) != NULL || !*phoenix_runtime_error())
        return 2;
    if (phoenix_runtime_frame(NULL, NULL, NULL, NULL, NULL, "{}", NULL) != -1 ||
        strstr(phoenix_runtime_error(), "runtime") == NULL)
        return 3;
    if (phoenix_runtime_status(NULL, NULL, 0) != 0)
        return 4;
    phoenix_runtime_destroy(NULL);
    puts("C API compile/link/error contracts passed");
    return 0;
}
