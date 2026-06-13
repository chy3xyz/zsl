const std = @import("std");
const vcl = @import("zsl").vcl;

pub fn main() !void {
    std.debug.print("OpenCL/VCL backend stub demonstration\n", .{});
    const allocator = std.heap.page_allocator;

    // Device creation is a stub.
    var device = vcl.VclDevice.init(allocator) catch |err| {
        std.debug.print("VclDevice.init: {s}\n", .{@errorName(err)});
        return;
    };
    device.deinit();

    // Buffer creation is a stub.
    const Buffer = vcl.VclBuffer(f64);
    var buffer = Buffer.init(allocator, 4) catch |err| {
        std.debug.print("VclBuffer(f64).init: {s}\n", .{@errorName(err)});
        return;
    };
    defer buffer.deinit();

    // Compute backend is a stub.
    var backend = vcl.VCLBackend.init(allocator);
    defer backend.deinit();

    std.debug.print("Backend name: {s}\n", .{backend.name()});
    std.debug.print("Supports gemm: {}\n", .{backend.supports("gemm")});
    std.debug.print("Supports conv2d: {}\n", .{backend.supports("conv2d")});

    const cb = backend.backend();
    var a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var c = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    cb.gemm(allocator, &a, &a, &c, 2, 2, 2) catch |err| {
        std.debug.print("VCLBackend.gemm: {s}\n", .{@errorName(err)});
    };

    std.debug.print("OpenCL/VCL stub demonstration complete\n", .{});
}
