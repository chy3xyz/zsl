const std = @import("std");
const types = @import("types.zig");

/// Stub Vulkan device.
///
/// Real Vulkan initialization is not implemented. All functions that would
/// interact with the Vulkan loader return `error.NotImplemented` and print a
/// diagnostic message via `std.debug.print`.
pub const VulkanDevice = struct {
    allocator: std.mem.Allocator,

    /// Attempt to create a Vulkan device.
    ///
    /// This stub always prints a message and returns `error.NotImplemented`.
    pub fn init(allocator: std.mem.Allocator) types.VulkanError!VulkanDevice {
        _ = allocator;
        std.debug.print("Vulkan not available: device initialization is a stub\n", .{});
        return error.NotImplemented;
    }

    /// Release the Vulkan device.
    ///
    /// This stub only prints a diagnostic message.
    pub fn deinit(self: *VulkanDevice) void {
        _ = self;
        std.debug.print("Vulkan not available: device deinitialization is a stub\n", .{});
    }
};

test "VulkanDevice.init returns NotImplemented" {
    try std.testing.expectError(
        error.NotImplemented,
        VulkanDevice.init(std.testing.allocator),
    );
}
