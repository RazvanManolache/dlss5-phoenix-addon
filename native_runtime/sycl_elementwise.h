#pragma once
// Keep pointwise SYCL kernels independent of reduction scratch arrays and
// matrix loops. Each specialization fixes the operation kind before device JIT.
template <unsigned Kind>
sycl::event sycl_unary(sycl::queue& queue, phx::Operation operation, uint64_t count) {
    return queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
        auto specialized = operation;
        specialized.flags = (operation.flags & ~255u) | Kind;
        phx::Kernel<GpuMath>{specialized}.unary(uint64_t(index[0]));
    });
}
sycl::event sycl_elementwise(sycl::queue& queue, phx::Operation operation, uint64_t count) {
    switch (operation.flags & 255u) {
    case phx::QuantizeFp8:
        return sycl_unary<phx::QuantizeFp8>(queue, operation, count);
    case phx::ActivationGate:
        return sycl_unary<phx::ActivationGate>(queue, operation, count);
    case phx::RoundHalf:
        return sycl_unary<phx::RoundHalf>(queue, operation, count);
    case phx::FloatToHalf:
        return sycl_unary<phx::FloatToHalf>(queue, operation, count);
    case phx::CopyValues:
        return sycl_unary<phx::CopyValues>(queue, operation, count);
    case phx::ScaledResidual:
        return sycl_unary<phx::ScaledResidual>(queue, operation, count);
    case phx::ScaleHalf:
        return sycl_unary<phx::ScaleHalf>(queue, operation, count);
    case phx::GatherWindows:
        return sycl_unary<phx::GatherWindows>(queue, operation, count);
    case phx::ScatterWindows:
        return sycl_unary<phx::ScatterWindows>(queue, operation, count);
    case phx::AddPositionBias:
        return sycl_unary<phx::AddPositionBias>(queue, operation, count);
    case phx::SplitQkvHeads:
        return sycl_unary<phx::SplitQkvHeads>(queue, operation, count);
    case phx::MergeAttentionHeads:
        return sycl_unary<phx::MergeAttentionHeads>(queue, operation, count);
    case phx::AveragePool:
        return sycl_unary<phx::AveragePool>(queue, operation, count);
    case phx::NearestUpsample:
        return sycl_unary<phx::NearestUpsample>(queue, operation, count);
    case phx::ChannelGain:
        return sycl_unary<phx::ChannelGain>(queue, operation, count);
    case phx::AddInputs:
        return sycl_unary<phx::AddInputs>(queue, operation, count);
    case phx::SpatialPad:
        return sycl_unary<phx::SpatialPad>(queue, operation, count);
    case phx::GateFp8ToHalf:
        return sycl_unary<phx::GateFp8ToHalf>(queue, operation, count);
    case phx::Fp8ToHalf:
        return sycl_unary<phx::Fp8ToHalf>(queue, operation, count);
    case phx::GateToHalf:
        return sycl_unary<phx::GateToHalf>(queue, operation, count);
    case phx::UpsampleBlendAndPublish:
        return sycl_unary<phx::UpsampleBlendAndPublish>(queue, operation, count);
    case phx::UpsampleResidual:
        return sycl_unary<phx::UpsampleResidual>(queue, operation, count);
    case phx::PoolAndPublish:
        return sycl_unary<phx::PoolAndPublish>(queue, operation, count);
    case phx::GatherWindowBatch:
        return sycl_unary<phx::GatherWindowBatch>(queue, operation, count);
    case phx::ScatterWindowBatch:
        return sycl_unary<phx::ScatterWindowBatch>(queue, operation, count);
    case phx::ScatterWindowBatchAndPool:
        return sycl_unary<phx::ScatterWindowBatchAndPool>(queue, operation, count);
    case phx::GatherUpsampleBlendBatch:
        return sycl_unary<phx::GatherUpsampleBlendBatch>(queue, operation, count);
    default:
        return queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
            phx::Kernel<GpuMath>{operation}.unary(uint64_t(index[0]));
        });
    }
}
