/// OpenCL/VCL compute backend stub.
///
/// This module exposes the public types needed by the compute dispatch layer.
/// All real OpenCL driver calls are stubbed out and return
/// `error.NotImplemented` because the zsl build does not link against OpenCL
/// by default.
const device = @import("vcl/device.zig");
const buffer = @import("vcl/buffer.zig");
const types = @import("vcl/types.zig");

pub const VclDevice = device.VclDevice;
pub const VclBuffer = buffer.VclBuffer;
pub const VclError = types.VclError;
pub const DeviceType = types.DeviceType;
pub const MemFlags = types.MemFlags;

pub const compute = @import("vcl/compute/backend.zig");
pub const VCLBackend = compute.VCLBackend;

// Alias matching the VSL top-level export `pub type Device = Device`.
pub const Device = VclDevice;

test {
    _ = device;
    _ = buffer;
    _ = types;
    _ = compute;
}
