const std = @import("std");
const cuda = @import("zsl").cuda;

pub fn main() !void {
    std.debug.print("=== CUDA Backend Stub Demo ===\n", .{});

    // Create a stub CUDA backend. No GPU is actually initialized.
    var backend = cuda.CUDABackend.init(std.heap.page_allocator);
    defer backend.deinit();

    std.debug.print("Backend name: {s}\n", .{backend.name()});
    std.debug.print("Supports gemm: {}\n", .{backend.supports("gemm")});
    std.debug.print("Supports conv2d: {}\n", .{backend.supports("conv2d")});

    // Obtain the backend-agnostic interface. All operations return
    // error.NotImplemented in the stub.
    const cb = backend.backend();

    var a = [_]f64{1.0};
    var b = [_]f64{2.0};
    var c = [_]f64{0.0};

    const result = cb.gemm(std.heap.page_allocator, &a, &b, &c, 1, 1, 1);
    if (result) |_| {
        std.debug.print("Unexpected success from CUDA gemm\n", .{});
    } else |err| {
        std.debug.print("CUDA gemm returned expected error: {s}\n", .{@errorName(err)});
    }

    // Device and memory stubs behave the same way.
    var dev = cuda.CudaDevice.init(std.heap.page_allocator);
    defer dev.deinit();

    const Mem = cuda.CudaMemory(f64);
    const mem = Mem.init(std.heap.page_allocator, 16);
    if (mem) |_| {
        std.debug.print("Unexpected success from CUDA memory allocation\n", .{});
    } else |err| {
        std.debug.print("CUDA memory allocation returned expected error: {s}\n", .{@errorName(err)});
    }

    std.debug.print("=== CUDA stub demo complete ===\n", .{});
}
