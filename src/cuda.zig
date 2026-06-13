/// CUDA compute backend stub.
///
/// This module exposes the public types needed by the compute dispatch layer.
/// All real CUDA driver/runtime calls are stubbed out and return
/// `error.NotImplemented` because the zsl build does not link against the
/// CUDA toolkit by default.
const device = @import("cuda/device.zig");
const memory = @import("cuda/memory.zig");
const types = @import("cuda/types.zig");

pub const CudaDevice = device.CudaDevice;
pub const CudaMemory = memory.CudaMemory;
pub const CudaContext = types.CudaContext;
pub const CublasHandle = types.CublasHandle;
pub const CudnnHandle = types.CudnnHandle;
pub const CudaStream = types.CudaStream;
pub const cuda_supported_ops = types.cuda_supported_ops;

pub const compute = @import("cuda/compute/backend.zig");
pub const CUDABackend = compute.CUDABackend;

// Alias matching the VSL top-level export `pub type Device = CudaDevice`.
pub const Device = CudaDevice;

test {
    _ = device;
    _ = memory;
    _ = types;
    _ = compute;
}
