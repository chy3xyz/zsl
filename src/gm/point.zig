const std = @import("std");
const util = @import("../util.zig");

/// A generic 3D point with Cartesian coordinates.
pub fn Point(comptime T: type) type {
    if (!util.isFloat(T)) {
        @compileError("Point requires a floating-point type");
    }

    return struct {
        x: T,
        y: T,
        z: T,

        const Self = @This();

        /// Create a new point in 3D space.
        pub fn new(x: T, y: T, z: T) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        /// Create a new point in 2D space (z is set to zero).
        pub fn new_2d(x: T, y: T) Self {
            return .{ .x = x, .y = y, .z = 0 };
        }

        /// Return an independent copy of this point.
        pub fn clone(self: Self) Self {
            return self;
        }

        /// Return a new point displaced by `dx`, `dy`, `dz`.
        pub fn displace(self: Self, dx: T, dy: T, dz: T) Self {
            return .{
                .x = self.x + dx,
                .y = self.y + dy,
                .z = self.z + dz,
            };
        }
    };
}

/// Squared Euclidean distance between `a` and `b` in `dim` dimensions.
fn distance_squared_dim(comptime T: type, comptime dim: u2, a: Point(T), b: Point(T)) T {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    var d = dx * dx + dy * dy;
    if (dim == 3) {
        const dz = a.z - b.z;
        d += dz * dz;
    }
    return d;
}

/// 3D Euclidean distance between `a` and `b`.
pub fn distance(comptime T: type, a: Point(T), b: Point(T)) T {
    return @sqrt(distance_squared(T, a, b));
}

/// 2D Euclidean distance between `a` and `b` (ignoring z).
pub fn distance_2d(comptime T: type, a: Point(T), b: Point(T)) T {
    return @sqrt(distance_squared_dim(T, 2, a, b));
}

/// Squared 3D Euclidean distance between `a` and `b`.
pub fn distance_squared(comptime T: type, a: Point(T), b: Point(T)) T {
    return distance_squared_dim(T, 3, a, b);
}

test "Point constructors and displacement" {
    const P = Point(f64);
    const a = P.new(1.0, 2.0, 3.0);
    const b = P.new_2d(1.0, 2.0);
    const c = a.displace(1.0, -1.0, 2.0);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), c.x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), c.y, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), c.z, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), b.z, 1e-12);

    const d = a.clone();
    try std.testing.expectApproxEqAbs(a.x, d.x, 1e-12);
    try std.testing.expectApproxEqAbs(a.y, d.y, 1e-12);
    try std.testing.expectApproxEqAbs(a.z, d.z, 1e-12);
}

test "Point distances" {
    const P = Point(f64);
    const a = P.new(0.0, 0.0, 0.0);
    const b = P.new(3.0, 4.0, 0.0);
    const c = P.new(1.0, 2.0, 2.0);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), distance(f64, a, b), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), distance_2d(f64, a, b), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), distance_squared(f64, a, b), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), distance(f64, a, c), 1e-12);
}
