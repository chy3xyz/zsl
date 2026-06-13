const std = @import("std");
const vulkan = @import("zsl").vulkan;

pub fn main() !void {
    std.debug.print("Vulkan backend stub demonstration\n", .{});

    const allocator = std.heap.page_allocator;

    // Device creation is a stub.
    var device = vulkan.VulkanDevice.init(allocator) catch |err| {
        std.debug.print("VulkanDevice.init: {s}\n", .{@errorName(err)});
        return;
    };
    device.deinit();

    // Buffer creation is a stub.
    const Buffer = vulkan.VulkanBuffer(f64);
    var buffer = Buffer.init(allocator, 4) catch |err| {
        std.debug.print("VulkanBuffer(f64).init: {s}\n", .{@errorName(err)});
        return;
    };
    defer buffer.deinit();

    // Compute backend is a stub.
    var backend = vulkan.VulkanBackend.init(allocator);
    defer backend.deinit();

    const cb = backend.backend();
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var c = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    cb.gemm(allocator, &a, &a, &c, 2, 2, 2) catch |err| {
        std.debug.print("VulkanBackend.gemm: {s}\n", .{@errorName(err)});
    };

    std.debug.print("Vulkan stub demonstration complete\n", .{});
}
