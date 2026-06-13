const std = @import("std");
const types = @import("types.zig");

/// Stub Vulkan buffer for a given element type.
///
/// Real GPU memory allocation, upload, and download are not implemented.
/// All functions return `error.NotImplemented` and print a diagnostic message.
pub fn VulkanBuffer(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        capacity: usize,

        /// Attempt to create a Vulkan buffer.
        pub fn init(allocator: std.mem.Allocator, capacity: usize) types.VulkanError!Self {
            _ = allocator;
            _ = capacity;
            std.debug.print("Vulkan not available: buffer initialization is a stub\n", .{});
            return error.NotImplemented;
        }

        /// Release the Vulkan buffer.
        pub fn deinit(self: *Self) void {
            _ = self;
            std.debug.print("Vulkan not available: buffer deinitialization is a stub\n", .{});
        }

        /// Upload host data to the Vulkan buffer.
        pub fn upload(self: *Self, data: []const T) types.VulkanError!void {
            _ = self;
            _ = data;
            std.debug.print("Vulkan not available: buffer upload is a stub\n", .{});
            return error.NotImplemented;
        }

        /// Download data from the Vulkan buffer to host memory.
        pub fn download(self: *Self, out: []T) types.VulkanError!void {
            _ = self;
            _ = out;
            std.debug.print("Vulkan not available: buffer download is a stub\n", .{});
            return error.NotImplemented;
        }
    };
}

test "VulkanBuffer operations return NotImplemented" {
    const Buf = VulkanBuffer(f64);
    try std.testing.expectError(error.NotImplemented, Buf.init(std.testing.allocator, 4));
}
