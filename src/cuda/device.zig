const std = @import("std");
const types = @import("types.zig");

/// `CudaDevice` is a stub representation of a CUDA device.
///
/// A real implementation would wrap the CUDA driver/runtime API, cuBLAS and
/// cuDNN handles. The stub only stores placeholder state and prints a message
/// on initialization; all functions that would talk to the GPU return
/// `error.NotImplemented`.
pub const CudaDevice = struct {
    allocator: std.mem.Allocator,
    device_id: usize = 0,
    name: []const u8 = "CUDA stub device",
    handle: usize = 0,
    ctx: types.CudaContext = 0,
    cublas: types.CublasHandle = 0,
    cudnn: types.CudnnHandle = 0,
    stream: types.CudaStream = 0,

    const Self = @This();

    /// Initialize a stub CUDA device.
    pub fn init(allocator: std.mem.Allocator) Self {
        std.debug.print("CUDA runtime not available; CudaDevice is a stub\n", .{});
        return .{
            .allocator = allocator,
        };
    }

    /// Release any resources held by the device. For the stub this is a no-op.
    pub fn deinit(self: *Self) void {
        self.* = undefined;
    }

    /// Initialize cuBLAS/cuDNN handles for this device.
    pub fn initHandles(self: *Self) error{NotImplemented}!void {
        _ = self;
        return error.NotImplemented;
    }

    /// Release cuBLAS/cuDNN handles and the CUDA context.
    pub fn release(self: *Self) error{NotImplemented}!void {
        _ = self;
        return error.NotImplemented;
    }

    /// Return the number of available CUDA devices.
    pub fn getDeviceCount() error{NotImplemented}!usize {
        return error.NotImplemented;
    }

    /// Return the CUDA device with the given ordinal.
    pub fn getDevice(index: usize) error{NotImplemented}!Self {
        _ = index;
        return error.NotImplemented;
    }

    /// Return the first available CUDA device.
    pub fn getDefaultDevice() error{NotImplemented}!Self {
        return error.NotImplemented;
    }
};

test "CudaDevice stub lifecycle" {
    var dev = CudaDevice.init(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), dev.device_id);
    dev.deinit();
}

test "CudaDevice GPU queries return NotImplemented" {
    try std.testing.expectError(error.NotImplemented, CudaDevice.getDeviceCount());
    try std.testing.expectError(error.NotImplemented, CudaDevice.getDevice(0));
    try std.testing.expectError(error.NotImplemented, CudaDevice.getDefaultDevice());

    var dev = CudaDevice.init(std.testing.allocator);
    defer dev.deinit();
    try std.testing.expectError(error.NotImplemented, dev.initHandles());
    try std.testing.expectError(error.NotImplemented, dev.release());
}
