const std = @import("std");
const ComputeBackend = @import("../../compute/backend.zig").ComputeBackend;
const device_module = @import("../device.zig");
const types = @import("../types.zig");

/// CUDA compute backend stub.
///
/// Implements the `zsl.compute.ComputeBackend` interface using placeholder
/// device/memory types. No CUDA runtime is linked, so every operation prints a
/// diagnostic and returns `error.NotImplemented`.
pub const CUDABackend = struct {
    allocator: std.mem.Allocator,
    device: device_module.CudaDevice,

    const Self = @This();

    /// Create a CUDA compute backend stub.
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .device = device_module.CudaDevice.init(allocator),
        };
    }

    /// Release any backend resources.
    pub fn deinit(self: *Self) void {
        self.device.deinit();
    }

    /// Return the backend identifier.
    pub fn name(self: Self) []const u8 {
        _ = self;
        return "cuda";
    }

    /// Return `true` when `op` is in the set of operations advertised by the
    /// CUDA backend.
    pub fn supports(self: Self, op: []const u8) bool {
        _ = self;
        for (types.cuda_supported_ops) |supported| {
            if (std.mem.eql(u8, supported, op)) return true;
        }
        return false;
    }

    /// Return a `ComputeBackend` interface backed by this CUDA stub.
    pub fn backend(self: *Self) ComputeBackend {
        return .{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = ComputeBackend.VTable{
        .gemm = gemmImpl,
        .gemv = gemvImpl,
        .relu = reluImpl,
        .sigmoid = sigmoidImpl,
        .tanh = tanhImpl,
        .softmax = softmaxImpl,
        .layernorm = layernormImpl,
    };

    fn gemmImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        a: []const f64,
        b: []const f64,
        c: []f64,
        m: usize,
        n: usize,
        k: usize,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = a;
        _ = b;
        _ = c;
        _ = m;
        _ = n;
        _ = k;
        std.debug.print("CUDA not available: gemm is a stub\n", .{});
        return error.NotImplemented;
    }

    fn gemvImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        a: []const f64,
        x: []const f64,
        y: []f64,
        m: usize,
        n: usize,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = a;
        _ = x;
        _ = y;
        _ = m;
        _ = n;
        std.debug.print("CUDA not available: gemv is a stub\n", .{});
        return error.NotImplemented;
    }

    fn reluImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = x;
        std.debug.print("CUDA not available: relu is a stub\n", .{});
        return error.NotImplemented;
    }

    fn sigmoidImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = x;
        std.debug.print("CUDA not available: sigmoid is a stub\n", .{});
        return error.NotImplemented;
    }

    fn tanhImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = x;
        std.debug.print("CUDA not available: tanh is a stub\n", .{});
        return error.NotImplemented;
    }

    fn softmaxImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
        rows: usize,
        cols: usize,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = x;
        _ = rows;
        _ = cols;
        std.debug.print("CUDA not available: softmax is a stub\n", .{});
        return error.NotImplemented;
    }

    fn layernormImpl(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        x: []f64,
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) ComputeBackend.Error!void {
        _ = ctx;
        _ = allocator;
        _ = x;
        _ = rows;
        _ = cols;
        _ = epsilon;
        std.debug.print("CUDA not available: layernorm is a stub\n", .{});
        return error.NotImplemented;
    }
};

test "CUDABackend name and supports" {
    var cuda_backend = CUDABackend.init(std.testing.allocator);
    defer cuda_backend.deinit();

    try std.testing.expectEqualStrings("cuda", cuda_backend.name());
    try std.testing.expect(cuda_backend.supports("gemm"));
    try std.testing.expect(cuda_backend.supports("softmax"));
    try std.testing.expect(!cuda_backend.supports("conv2d"));
}

test "CUDABackend compute operations return NotImplemented" {
    var cuda_backend = CUDABackend.init(std.testing.allocator);
    defer cuda_backend.deinit();

    const cb = cuda_backend.backend();
    var a = [_]f64{1.0};
    var b = [_]f64{2.0};
    var c = [_]f64{0.0};

    try std.testing.expectError(error.NotImplemented, cb.gemm(std.testing.allocator, &a, &b, &c, 1, 1, 1));
    try std.testing.expectError(error.NotImplemented, cb.gemv(std.testing.allocator, &a, &b, &c, 1, 1));
    try std.testing.expectError(error.NotImplemented, cb.relu(std.testing.allocator, &c));
    try std.testing.expectError(error.NotImplemented, cb.sigmoid(std.testing.allocator, &c));
    try std.testing.expectError(error.NotImplemented, cb.tanh(std.testing.allocator, &c));
    try std.testing.expectError(error.NotImplemented, cb.softmax(std.testing.allocator, &c, 1, 1));
    try std.testing.expectError(error.NotImplemented, cb.layernorm(std.testing.allocator, &c, 1, 1, 1e-8));
}
