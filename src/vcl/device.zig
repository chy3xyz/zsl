const std = @import("std");
const types = @import("types.zig");

/// Stub OpenCL/VCL device.
///
/// A real implementation would wrap platform enumeration, context creation, and
/// command queues. The stub stores only an allocator and prints a diagnostic on
/// initialization; all GPU-interacting functions return `error.NotImplemented`.
pub const VclDevice = struct {
    allocator: std.mem.Allocator,

    /// Attempt to create an OpenCL/VCL device.
    ///
    /// This stub always prints a message and returns `error.NotImplemented`.
    pub fn init(allocator: std.mem.Allocator) types.VclError!VclDevice {
        _ = allocator;
        std.debug.print("OpenCL/VCL not available: device initialization is a stub\n", .{});
        return error.NotImplemented;
    }

    /// Release the OpenCL/VCL device.
    ///
    /// This stub only prints a diagnostic message.
    pub fn deinit(self: *VclDevice) void {
        _ = self;
        std.debug.print("OpenCL/VCL not available: device deinitialization is a stub\n", .{});
    }

    /// Compile an OpenCL C program source for this device.
    pub fn add_program(self: *VclDevice, source: []const u8) types.VclError!void {
        _ = self;
        _ = source;
        std.debug.print("OpenCL/VCL not available: add_program is a stub\n", .{});
        return error.NotImplemented;
    }
};

test "VclDevice.init returns NotImplemented" {
    try std.testing.expectError(
        error.NotImplemented,
        VclDevice.init(std.testing.allocator),
    );
}
