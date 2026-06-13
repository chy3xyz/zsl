const std = @import("std");
const ComputeBackend = @import("../../compute/backend.zig").ComputeBackend;

/// OpenCL/VCL compute backend stub.
///
/// Implements the `zsl.compute.ComputeBackend` interface but does not perform
/// any real GPU work. Every operation prints a diagnostic and returns
/// `error.NotImplemented` so that callers can detect the missing backend at
/// runtime.
pub const VCLBackend = struct {
    allocator: std.mem.Allocator,

    /// Create an OpenCL/VCL compute backend stub.
    pub fn init(allocator: std.mem.Allocator) VCLBackend {
        return .{ .allocator = allocator };
    }

    /// Release any backend resources.
    ///
    /// The stub has no resources to release.
    pub fn deinit(self: *VCLBackend) void {
        _ = self;
    }

    /// Return the backend identifier.
    pub fn name(self: VCLBackend) []const u8 {
        _ = self;
        return "vcl";
    }

    /// Return `true` when `op` is in the set of operations advertised by the
    /// OpenCL/VCL backend.
    pub fn supports(self: VCLBackend, op: []const u8) bool {
        _ = self;
        const supported_ops = [_][]const u8{
            "gemm",
            "gemv",
            "relu",
            "sigmoid",
            "tanh",
            "softmax",
            "layernorm",
        };
        for (supported_ops) |supported| {
            if (std.mem.eql(u8, supported, op)) return true;
        }
        return false;
    }

    /// Return a `ComputeBackend` interface backed by this OpenCL/VCL stub.
    pub fn backend(self: *VCLBackend) ComputeBackend {
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
        std.debug.print("OpenCL/VCL not available: gemm is a stub\n", .{});
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
        std.debug.print("OpenCL/VCL not available: gemv is a stub\n", .{});
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
        std.debug.print("OpenCL/VCL not available: relu is a stub\n", .{});
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
        std.debug.print("OpenCL/VCL not available: sigmoid is a stub\n", .{});
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
        std.debug.print("OpenCL/VCL not available: tanh is a stub\n", .{});
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
        std.debug.print("OpenCL/VCL not available: softmax is a stub\n", .{});
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
        std.debug.print("OpenCL/VCL not available: layernorm is a stub\n", .{});
        return error.NotImplemented;
    }
};

test "VCLBackend name and supports" {
    var vcl_backend = VCLBackend.init(std.testing.allocator);
    defer vcl_backend.deinit();

    try std.testing.expectEqualStrings("vcl", vcl_backend.name());
    try std.testing.expect(vcl_backend.supports("gemm"));
    try std.testing.expect(vcl_backend.supports("softmax"));
    try std.testing.expect(!vcl_backend.supports("conv2d"));
}

test "VCLBackend compute operations return NotImplemented" {
    var vcl_backend = VCLBackend.init(std.testing.allocator);
    defer vcl_backend.deinit();

    const cb = vcl_backend.backend();
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
