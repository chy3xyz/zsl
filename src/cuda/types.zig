const std = @import("std");

/// Opaque handle types for CUDA driver/runtime resources.
/// In the stub backend these are just integers; a real implementation would
/// bind to the CUDA driver/runtime/cuBLAS/cuDNN handles.
pub const CudaContext = usize;
pub const CublasHandle = usize;
pub const CudnnHandle = usize;
pub const CudaStream = usize;

/// List of operation names advertised as supported by the CUDA backend.
/// Matches the operations defined by `zsl.compute.ComputeBackend`.
pub const cuda_supported_ops = [_][]const u8{
    "gemm",
    "gemv",
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "layernorm",
};

test "CUDA type aliases are usable" {
    const ctx: CudaContext = 0;
    const handle: CublasHandle = 0;
    const dnn: CudnnHandle = 0;
    const stream: CudaStream = 0;
    try std.testing.expectEqual(@as(CudaContext, 0), ctx);
    try std.testing.expectEqual(@as(CublasHandle, 0), handle);
    try std.testing.expectEqual(@as(CudnnHandle, 0), dnn);
    try std.testing.expectEqual(@as(CudaStream, 0), stream);
}
