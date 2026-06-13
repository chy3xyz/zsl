const std = @import("std");
const types = @import("types.zig");

/// Stub OpenCL/VCL buffer for a given element type.
///
/// Real device memory allocation, upload, and download are not implemented.
/// All functions return `error.NotImplemented` and print a diagnostic message.
pub fn VclBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        capacity: usize,

        /// Attempt to create an OpenCL/VCL buffer.
        pub fn init(allocator: std.mem.Allocator, capacity: usize) types.VclError!Self {
            _ = allocator;
            _ = capacity;
            std.debug.print("OpenCL/VCL not available: buffer initialization is a stub\n", .{});
            return error.NotImplemented;
        }

        /// Release the OpenCL/VCL buffer.
        pub fn deinit(self: *Self) void {
            _ = self;
            std.debug.print("OpenCL/VCL not available: buffer deinitialization is a stub\n", .{});
        }

        /// Upload host data to the OpenCL/VCL buffer.
        pub fn upload(self: *Self, data: []const T) types.VclError!void {
            _ = self;
            _ = data;
            std.debug.print("OpenCL/VCL not available: buffer upload is a stub\n", .{});
            return error.NotImplemented;
        }

        /// Download data from the OpenCL/VCL buffer to host memory.
        pub fn download(self: *Self, out: []T) types.VclError!void {
            _ = self;
            _ = out;
            std.debug.print("OpenCL/VCL not available: buffer download is a stub\n", .{});
            return error.NotImplemented;
        }
    };
}

test "VclBuffer operations return NotImplemented" {
    const Buf = VclBuffer(f64);
    try std.testing.expectError(error.NotImplemented, Buf.init(std.testing.allocator, 4));
}
