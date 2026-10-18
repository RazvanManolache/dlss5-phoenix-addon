// Compile as C, link the published ABI, and verify failures without a model.
#include "phoenix.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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
    phoenix_frame_v2 bounded = {0};
    bounded.struct_bytes = sizeof(bounded);
    bounded.version = PHOENIX_FRAME_VERSION_2;
    if (phoenix_runtime_frame_v2(NULL, &bounded) != -1 ||
        strstr(phoenix_runtime_error(), "runtime") == NULL)
        return 5;
    {
        size_t bytes = phoenix_runtime_capabilities(NULL, 0);
        char sentinel = 'x';
        char *text;
        if (bytes < 2 || phoenix_runtime_capabilities(&sentinel, 1) != bytes || sentinel != 'x')
            return 6;
        text = (char*)malloc(bytes);
        if (!text) return 7;
        if (phoenix_runtime_capabilities(text, bytes) != bytes || text[bytes-1] != '\0' ||
            strstr(text, "phoenix-experimental-rgb/1") == NULL) {
            free(text); return 8;
        }
        free(text);
    }
    puts("C API compile/link/error contracts passed");
    return 0;
}
