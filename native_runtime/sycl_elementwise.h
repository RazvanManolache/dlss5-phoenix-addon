#pragma once
// Keep pointwise SYCL kernels independent of reduction scratch arrays and
// matrix loops. Each specialization fixes the operation kind before device JIT.
template <unsigned Kind, class Math>
sycl::event sycl_unary(sycl::queue& queue, phx::Operation operation, uint64_t count) {
    return queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
        auto specialized = operation;
        specialized.flags = (operation.flags & ~255u) | Kind;
        phx::Kernel<Math>{specialized}.unary(uint64_t(index[0]));
    });
}
template <class Math>
sycl::event sycl_elementwise(sycl::queue& queue, phx::Operation operation, uint64_t count) {
    switch (operation.flags & 255u) {
    case phx::QuantizeFp8:
        return sycl_unary<phx::QuantizeFp8, Math>(queue, operation, count);
    case phx::ActivationGate:
        return sycl_unary<phx::ActivationGate, Math>(queue, operation, count);
    case phx::RoundHalf:
        return sycl_unary<phx::RoundHalf, Math>(queue, operation, count);
    case phx::FloatToHalf:
        return sycl_unary<phx::FloatToHalf, Math>(queue, operation, count);
    case phx::CopyValues:
        return sycl_unary<phx::CopyValues, Math>(queue, operation, count);
    case phx::ScaledResidual:
        return sycl_unary<phx::ScaledResidual, Math>(queue, operation, count);
    case phx::ScaleHalf:
        return sycl_unary<phx::ScaleHalf, Math>(queue, operation, count);
    case phx::GatherWindows:
        return sycl_unary<phx::GatherWindows, Math>(queue, operation, count);
    case phx::ScatterWindows:
        return sycl_unary<phx::ScatterWindows, Math>(queue, operation, count);
    case phx::AddPositionBias:
        return sycl_unary<phx::AddPositionBias, Math>(queue, operation, count);
    case phx::SplitQkvHeads:
        return sycl_unary<phx::SplitQkvHeads, Math>(queue, operation, count);
    case phx::MergeAttentionHeads:
        return sycl_unary<phx::MergeAttentionHeads, Math>(queue, operation, count);
    case phx::AveragePool:
        return sycl_unary<phx::AveragePool, Math>(queue, operation, count);
    case phx::NearestUpsample:
        return sycl_unary<phx::NearestUpsample, Math>(queue, operation, count);
    case phx::ChannelGain:
        return sycl_unary<phx::ChannelGain, Math>(queue, operation, count);
    case phx::AddInputs:
        return sycl_unary<phx::AddInputs, Math>(queue, operation, count);
    case phx::SpatialPad:
        return sycl_unary<phx::SpatialPad, Math>(queue, operation, count);
    case phx::GateFp8ToHalf:
        return sycl_unary<phx::GateFp8ToHalf, Math>(queue, operation, count);
    case phx::Fp8ToHalf:
        return sycl_unary<phx::Fp8ToHalf, Math>(queue, operation, count);
    case phx::GateToHalf:
        return sycl_unary<phx::GateToHalf, Math>(queue, operation, count);
    case phx::UpsampleBlendAndPublish:
        return sycl_unary<phx::UpsampleBlendAndPublish, Math>(queue, operation, count);
    case phx::UpsampleResidual:
        return sycl_unary<phx::UpsampleResidual, Math>(queue, operation, count);
    case phx::PoolAndPublish:
        return sycl_unary<phx::PoolAndPublish, Math>(queue, operation, count);
    case phx::GatherWindowBatch:
        return sycl_unary<phx::GatherWindowBatch, Math>(queue, operation, count);
    case phx::ScatterWindowBatch:
        return sycl_unary<phx::ScatterWindowBatch, Math>(queue, operation, count);
    case phx::ScatterWindowBatchAndPool:
        return sycl_unary<phx::ScatterWindowBatchAndPool, Math>(queue, operation, count);
    case phx::GatherUpsampleBlendBatch:
        return sycl_unary<phx::GatherUpsampleBlendBatch, Math>(queue, operation, count);
    default:
        return queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
            phx::Kernel<Math>{operation}.unary(uint64_t(index[0]));
        });
    }
}
