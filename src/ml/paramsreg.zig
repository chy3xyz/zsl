const std = @import("std");
const Error = @import("../errors.zig").Error;

/// `ParamsReg(T)` holds the parameters of a regression model.
///
///   theta  -- weight vector [nb_features]
///   bias   -- intercept term
///   lambda -- L2 regularization strength
///   degree -- polynomial degree (for polynomial regression)
pub fn ParamsReg(comptime T: type) type {
    return struct {
        theta: []T,
        bias: T,
        lambda: T,
        degree: i32,
        // backup fields
        bkp_theta: []T,
        bkp_bias: T,
        bkp_lambda: T,
        bkp_degree: i32,

        const Self = @This();

        /// Allocates a new parameter object with `nb_features` weights.
        pub fn init(allocator: std.mem.Allocator, nb_features: usize) Error!Self {
            if (nb_features == 0) return error.InvalidDimension;

            const theta = try allocator.alloc(T, nb_features);
            errdefer allocator.free(theta);
            const bkp_theta = try allocator.alloc(T, nb_features);
            errdefer allocator.free(bkp_theta);

            @memset(theta, 0);
            @memset(bkp_theta, 0);

            return .{
                .theta = theta,
                .bias = 0,
                .lambda = 0,
                .degree = 1,
                .bkp_theta = bkp_theta,
                .bkp_bias = 0,
                .bkp_lambda = 0,
                .bkp_degree = 1,
            };
        }

        /// Sets the whole weight vector and the bias.
        pub fn set_params(self: *Self, theta: []const T, b: T) Error!void {
            if (theta.len != self.theta.len) return error.ShapeMismatch;
            @memcpy(self.theta, theta);
            self.bias = b;
        }

        /// Sets a single parameter. Use negative indices for the bias.
        pub fn set_param(self: *Self, i: isize, value: T) Error!void {
            if (i < 0) {
                self.bias = value;
                return;
            }
            const idx: usize = @intCast(i);
            if (idx >= self.theta.len) return error.IndexOutOfBounds;
            self.theta[idx] = value;
        }

        /// Gets a single parameter. Use negative indices for the bias.
        pub fn get_param(self: *Self, i: isize) Error!T {
            if (i < 0) return self.bias;
            const idx: usize = @intCast(i);
            if (idx >= self.theta.len) return error.IndexOutOfBounds;
            return self.theta[idx];
        }

        /// Sets `theta[i]`.
        pub fn set_theta(self: *Self, i: usize, value: T) Error!void {
            if (i >= self.theta.len) return error.IndexOutOfBounds;
            self.theta[i] = value;
        }

        /// Gets `theta[i]`.
        pub fn get_theta(self: *Self, i: usize) Error!T {
            if (i >= self.theta.len) return error.IndexOutOfBounds;
            return self.theta[i];
        }

        /// Sets the bias.
        pub fn set_bias(self: *Self, b: T) void {
            self.bias = b;
        }

        /// Gets the bias.
        pub fn get_bias(self: *Self) T {
            return self.bias;
        }

        /// Sets the regularization strength.
        pub fn set_lambda(self: *Self, lambda: T) void {
            self.lambda = lambda;
        }

        /// Gets the regularization strength.
        pub fn get_lambda(self: *Self) T {
            return self.lambda;
        }

        /// Sets the polynomial degree.
        pub fn set_degree(self: *Self, degree: i32) void {
            self.degree = degree;
        }

        /// Gets the polynomial degree.
        pub fn get_degree(self: *Self) i32 {
            return self.degree;
        }

        /// Saves a copy of the current parameters.
        pub fn backup(self: *Self) void {
            @memcpy(self.bkp_theta, self.theta);
            self.bkp_bias = self.bias;
            self.bkp_lambda = self.lambda;
            self.bkp_degree = self.degree;
        }

        /// Restores parameters from the backup copy.
        pub fn restore(self: *Self) void {
            @memcpy(self.theta, self.bkp_theta);
            self.bias = self.bkp_bias;
            self.lambda = self.bkp_lambda;
            self.degree = self.bkp_degree;
        }

        /// Releases all owned memory.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.theta);
            allocator.free(self.bkp_theta);
            self.theta = &[_]T{};
            self.bkp_theta = &[_]T{};
        }
    };
}

test "ParamsReg get/set and backup/restore" {
    const T = f64;
    var p = try ParamsReg(T).init(std.testing.allocator, 2);
    defer p.deinit(std.testing.allocator);

    try p.set_params(&[_]T{ 1.0, 2.0 }, 3.0);
    try std.testing.expectApproxEqAbs(1.0, try p.get_theta(0), 1e-12);
    try std.testing.expectApproxEqAbs(2.0, try p.get_theta(1), 1e-12);
    try std.testing.expectApproxEqAbs(3.0, p.get_bias(), 1e-12);

    try p.set_param(0, 5.0);
    try p.set_param(-1, 7.0);
    try std.testing.expectApproxEqAbs(5.0, try p.get_param(0), 1e-12);
    try std.testing.expectApproxEqAbs(7.0, try p.get_param(-1), 1e-12);

    p.set_lambda(0.5);
    p.set_degree(2);
    try std.testing.expectApproxEqAbs(0.5, p.get_lambda(), 1e-12);
    try std.testing.expectEqual(@as(i32, 2), p.get_degree());

    p.backup();
    try p.set_params(&[_]T{ 0.0, 0.0 }, 0.0);
    p.set_lambda(0.0);
    p.restore();
    try std.testing.expectApproxEqAbs(5.0, try p.get_theta(0), 1e-12);
    try std.testing.expectApproxEqAbs(2.0, try p.get_theta(1), 1e-12);
    try std.testing.expectApproxEqAbs(7.0, p.get_bias(), 1e-12);
    try std.testing.expectApproxEqAbs(0.5, p.get_lambda(), 1e-12);
    try std.testing.expectEqual(@as(i32, 2), p.get_degree());
}
