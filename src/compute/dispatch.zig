const std = @import("std");
const Error = @import("../errors.zig").Error;
const context = @import("context.zig");
const backend = @import("backend.zig");
const cpu_backend = @import("backend_cpu.zig");

/// High-level dispatcher that owns a compute context and the active backend
/// implementation. Operations are forwarded to the backend selected by the
/// caller.
pub const ComputeDispatch = struct {
    ctx: context.ComputeContext,
    backend_impl: backend.ComputeBackend,
    cpu: cpu_backend.CpuBackend,

    /// Create a dispatcher for the requested backend. The CPU backend is used
    /// when `backend_kind` is `.cpu` or `.auto`.
    pub fn init(allocator: std.mem.Allocator, backend_kind: context.Backend) Error!ComputeDispatch {
        const ctx = context.ComputeContext.init(allocator, backend_kind);
        var self = ComputeDispatch{
            .ctx = ctx,
            .backend_impl = undefined,
            .cpu = .{},
        };
        switch (backend_kind) {
            .cpu, .auto => self.backend_impl = self.cpu.backend(),
            else => return error.NotImplemented,
        }
        return self;
    }

    /// Release dispatcher resources.
    pub fn deinit(self: *ComputeDispatch) void {
        self.ctx.deinit();
    }

    pub fn gemm(
        self: *ComputeDispatch,
        a: []const f64,
        b: []const f64,
        c: []f64,
        m: usize,
        n: usize,
        k: usize,
    ) Error!void {
        return self.backend_impl.gemm(self.ctx.allocator, a, b, c, m, n, k);
    }

    pub fn gemv(
        self: *ComputeDispatch,
        a: []const f64,
        x: []const f64,
        y: []f64,
        m: usize,
        n: usize,
    ) Error!void {
        return self.backend_impl.gemv(self.ctx.allocator, a, x, y, m, n);
    }

    pub fn relu(self: *ComputeDispatch, x: []f64) Error!void {
        return self.backend_impl.relu(self.ctx.allocator, x);
    }

    pub fn sigmoid(self: *ComputeDispatch, x: []f64) Error!void {
        return self.backend_impl.sigmoid(self.ctx.allocator, x);
    }

    pub fn tanh(self: *ComputeDispatch, x: []f64) Error!void {
        return self.backend_impl.tanh(self.ctx.allocator, x);
    }

    pub fn softmax(self: *ComputeDispatch, x: []f64, rows: usize, cols: usize) Error!void {
        return self.backend_impl.softmax(self.ctx.allocator, x, rows, cols);
    }

    pub fn layernorm(self: *ComputeDispatch, x: []f64, rows: usize, cols: usize, epsilon: f64) Error!void {
        return self.backend_impl.layernorm(self.ctx.allocator, x, rows, cols, epsilon);
    }
};

test "ComputeDispatch CPU gemm" {
    var dispatch = try ComputeDispatch.init(std.testing.allocator, .cpu);
    defer dispatch.deinit();

    const a = &[_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const b = &[_]f64{ 1.0, 0.0, 0.0, 1.0 };
    var c = [_]f64{ 0.0, 0.0, 0.0, 0.0 };

    try dispatch.gemm(a, b, &c, 2, 2, 2);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), c[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), c[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), c[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), c[3], 1e-12);
}

test "ComputeDispatch auto selects CPU" {
    var dispatch = try ComputeDispatch.init(std.testing.allocator, .auto);
    defer dispatch.deinit();
    try std.testing.expect(dispatch.ctx.backend == .auto);
}

test "ComputeDispatch rejects unimplemented backends" {
    try std.testing.expectError(error.NotImplemented, ComputeDispatch.init(std.testing.allocator, .vulkan));
}
