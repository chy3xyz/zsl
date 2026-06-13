const std = @import("std");

/// `CudaMemory(T)` is a stub helper for managing GPU device memory.
///
/// A real implementation would call `cudaMalloc`, `cudaFree` and `cudaMemcpy`.
/// The stub stores the intended element count but never allocates device
/// memory; `upload`, `download` and allocation all return `error.NotImplemented`.
pub fn CudaMemory(comptime T: type) type {
    return struct {
        ptr: ?*anyopaque,
        count: usize,

        const Self = @This();

        /// Allocate GPU memory for `count` elements of type `T`.
        pub fn init(allocator: std.mem.Allocator, count: usize) error{NotImplemented}!Self {
            _ = allocator;
            _ = count;
            std.debug.print("CUDA memory allocation not available; CudaMemory is a stub\n", .{});
            return error.NotImplemented;
        }

        /// Release the GPU memory. For the stub this is a no-op.
        pub fn deinit(self: *Self) void {
            self.* = .{
                .ptr = null,
                .count = 0,
            };
        }

        /// Copy data from host to device.
        pub fn upload(self: *Self, data: []const T) error{NotImplemented}!void {
            _ = self;
            _ = data;
            return error.NotImplemented;
        }

        /// Copy data from device to host.
        pub fn download(self: Self, out: []T) error{NotImplemented}!void {
            _ = self;
            _ = out;
            return error.NotImplemented;
        }

        /// Alias for deinit.
        pub fn free(self: *Self) void {
            self.deinit();
        }
    };
}

test "CudaMemory allocation returns NotImplemented" {
    const Mem = CudaMemory(f64);
    try std.testing.expectError(error.NotImplemented, Mem.init(std.testing.allocator, 16));
}

test "CudaMemory upload/download return NotImplemented" {
    const Mem = CudaMemory(f64);
    var mem = Mem{
        .ptr = null,
        .count = 0,
    };
    defer mem.deinit();

    const data = [_]f64{ 1.0, 2.0, 3.0 };
    var out = [_]f64{ 0.0, 0.0, 0.0 };
    try std.testing.expectError(error.NotImplemented, mem.upload(&data));
    try std.testing.expectError(error.NotImplemented, mem.download(&out));
}
