const std = @import("std");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;
const Vector = @import("la.zig").Vector;

pub fn axpy(comptime T: type, alpha: T, x: Vector(T), y: *Vector(T)) Error!void {
    _ = util.Float(T);
    _ = alpha;
    _ = x;
    _ = y;
    return error.ShapeMismatch;
}
