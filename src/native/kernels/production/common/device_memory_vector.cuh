// Specialized native memory-access helper for the selected kernel variant.
// Vector width, address space and half/FP8 packing distinguish this helper from its siblings.
// Do not merge variants solely because their function signatures match.

#include "async_memory_native_vectors.cuh"
#include "cached_vector_access.cuh"
