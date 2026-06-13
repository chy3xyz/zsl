const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Errors that can be returned by a compute backend.
pub const ComputeError = Error;

/// Backend-agnostic compute interface.
///
/// Concrete backends (CPU, Vulkan, CUDA, OpenCL) provide a vtable that
/// implements the supported operations. All dimensions are row-major.
pub const ComputeBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    /// Shorthand for the compute error set inside the interface namespace.
    pub const Error = ComputeError;

    pub const VTable = struct {
        gemm: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            a: []const f64,
            b: []const f64,
            c: []f64,
            m: usize,
            n: usize,
            k: usize,
        ) ComputeError!void,

        gemv: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            a: []const f64,
            x: []const f64,
            y: []f64,
            m: usize,
            n: usize,
        ) ComputeError!void,

        relu: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            x: []f64,
        ) ComputeError!void,

        sigmoid: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            x: []f64,
        ) ComputeError!void,

        tanh: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            x: []f64,
        ) ComputeError!void,

        softmax: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            x: []f64,
            rows: usize,
            cols: usize,
        ) ComputeError!void,

        layernorm: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            x: []f64,
            rows: usize,
            cols: usize,
            epsilon: f64,
        ) ComputeError!void,
    };

    pub fn gemm(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        a: []const f64,
        b: []const f64,
        c: []f64,
        m: usize,
        n: usize,
        k: usize,
    ) ComputeError!void {
        return self.vtable.gemm(self.ptr, allocator, a, b, c, m, n, k);
    }

    pub fn gemv(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        a: []const f64,
        x: []const f64,
        y: []f64,
        m: usize,
        n: usize,
    ) ComputeError!void {
        return self.vtable.gemv(self.ptr, allocator, a, x, y, m, n);
    }

    pub fn relu(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        x: []f64,
    ) ComputeError!void {
        return self.vtable.relu(self.ptr, allocator, x);
    }

    pub fn sigmoid(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        x: []f64,
    ) ComputeError!void {
        return self.vtable.sigmoid(self.ptr, allocator, x);
    }

    pub fn tanh(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        x: []f64,
    ) ComputeError!void {
        return self.vtable.tanh(self.ptr, allocator, x);
    }

    pub fn softmax(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        x: []f64,
        rows: usize,
        cols: usize,
    ) ComputeError!void {
        return self.vtable.softmax(self.ptr, allocator, x, rows, cols);
    }

    pub fn layernorm(
        self: ComputeBackend,
        allocator: std.mem.Allocator,
        x: []f64,
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) ComputeError!void {
        return self.vtable.layernorm(self.ptr, allocator, x, rows, cols, epsilon);
    }
};
