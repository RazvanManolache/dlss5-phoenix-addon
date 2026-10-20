// Aligned 16-byte feature/weight access. One vector instruction replaces four
// scalar requests while preserving the original cache modes and async helpers.
// Callers already satisfy check_vector_address's 16-byte alignment contract.
#include "async_memory.cuh"

#include "cached_vector_access.cuh"
