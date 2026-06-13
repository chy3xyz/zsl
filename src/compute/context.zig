const std = @import("std");

/// Supported compute backends.
pub const Backend = enum {
    auto,
    cpu,
    vulkan,
    cuda,
    opencl,
};

/// Lightweight container for backend selection and allocation state.
pub const ComputeContext = struct {
    backend: Backend,
    allocator: std.mem.Allocator,

    /// Create a new compute context.
    pub fn init(allocator: std.mem.Allocator, backend: Backend) ComputeContext {
        return .{
            .backend = backend,
            .allocator = allocator,
        };
    }

    /// Release the context. The context itself owns no allocations.
    pub fn deinit(self: *ComputeContext) void {
        self.* = undefined;
    }
};

test "ComputeContext stores backend and allocator" {
    var ctx = ComputeContext.init(std.testing.allocator, .cpu);
    defer ctx.deinit();
    try std.testing.expect(ctx.backend == .cpu);
    try std.testing.expect(ctx.allocator.ptr == std.testing.allocator.ptr);
}
