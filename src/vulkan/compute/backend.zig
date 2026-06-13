const std = @import("std");
const ComputeBackend = @import("../../compute/backend.zig").ComputeBackend;

/// Vulkan compute backend stub.
///
/// Implements the `zsl.compute.ComputeBackend` interface but does not perform
/// any real GPU work. Every operation prints a diagnostic and returns
/// `error.NotImplemented` so that callers can detect the missing backend at
/// runtime.
pub const VulkanBackend = struct {
    allocator: std.mem.Allocator,

    /// Create a Vulkan compute backend stub.
    pub fn init(allocator: std.mem.Allocator) VulkanBackend {
        return .{ .allocator = allocator };
    }

    /// Release any backend resources.
    ///
    /// The stub has no resources to release.
    pub fn deinit(self: *VulkanBackend) void {
        _ = self;
    }

    /// Return a `ComputeBackend` interface backed by this Vulkan stub.
    pub fn backend(self: *VulkanBackend) ComputeBackend {
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
        std.debug.print("Vulkan not available: gemm is a stub\n", .{});
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
        std.debug.print("Vulkan not available: gemv is a stub\n", .{});
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
        std.debug.print("Vulkan not available: relu is a stub\n", .{});
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
        std.debug.print("Vulkan not available: sigmoid is a stub\n", .{});
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
        std.debug.print("Vulkan not available: tanh is a stub\n", .{});
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
        std.debug.print("Vulkan not available: softmax is a stub\n", .{});
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
        std.debug.print("Vulkan not available: layernorm is a stub\n", .{});
        return error.NotImplemented;
    }
};

test "VulkanBackend compute operations return NotImplemented" {
    var vulkan_backend = VulkanBackend.init(std.testing.allocator);
    defer vulkan_backend.deinit();

    const cb = vulkan_backend.backend();
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
