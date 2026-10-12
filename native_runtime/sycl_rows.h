#pragma once
// Specialize the row family before device JIT, keeping its private reduction
// arrays out of unrelated kernels. CPU/GPU math policies remain independent.
template <unsigned Kind, class Math>
sycl::event sycl_row_kind(sycl::queue& queue, phx::Operation operation, uint64_t count) {
    return queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
        phx::Kernel<Math> math{operation};
        if constexpr (Kind == 0)
            math.norm(unsigned(index[0]));
        else if constexpr (Kind == 1)
            math.softmax(unsigned(index[0]));
        else
            math.global_attention_softmax(unsigned(index[0]));
    });
}
template <class Math>
sycl::event sycl_row(sycl::queue& queue, phx::Operation operation, uint64_t count) {
    switch (operation.flags & 255u) {
    case 0: // Cosine normalization of query/key vectors.
        return sycl_row_kind<0, Math>(queue, operation, count);
    case 1: // Normalize the 64 local-window attention scores.
        return sycl_row_kind<1, Math>(queue, operation, count);
    case 6: // Normalize a variable-length global-attention row.
        return sycl_row_kind<6, Math>(queue, operation, count);
    default:
        throw std::runtime_error("Unsupported specialized SYCL row operation");
    }
}
