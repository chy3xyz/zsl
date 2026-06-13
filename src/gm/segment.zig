const std = @import("std");
const point = @import("point.zig");
const util = @import("../util.zig");

/// A generic directed segment from point `a` to point `b`.
pub fn Segment(comptime T: type) type {
    if (!util.isFloat(T)) {
        @compileError("Segment requires a floating-point type");
    }

    return struct {
        a: point.Point(T),
        b: point.Point(T),

        const Self = @This();

        /// Create a new segment from two points.
        pub fn new(a: point.Point(T), b: point.Point(T)) Self {
            return .{ .a = a, .b = b };
        }

        /// Create a new segment from 2D coordinates (z is set to zero).
        pub fn new_2d(ax: T, ay: T, bx: T, by: T) Self {
            return .{
                .a = point.Point(T).new_2d(ax, ay),
                .b = point.Point(T).new_2d(bx, by),
            };
        }
    };
}

/// Shortest distance from point `p` to segment `seg` in `dim` dimensions.
fn distance_point_to_segment_dim(comptime T: type, comptime dim: u2, seg: Segment(T), p: point.Point(T)) T {
    const abx = seg.b.x - seg.a.x;
    const aby = seg.b.y - seg.a.y;
    const abz = if (dim == 3) seg.b.z - seg.a.z else 0;

    const apx = p.x - seg.a.x;
    const apy = p.y - seg.a.y;
    const apz = if (dim == 3) p.z - seg.a.z else 0;

    const ab2 = abx * abx + aby * aby + abz * abz;
    if (ab2 == 0) {
        return if (dim == 3)
            point.distance(T, p, seg.a)
        else
            point.distance_2d(T, p, seg.a);
    }

    const t = std.math.clamp(
        (apx * abx + apy * aby + apz * abz) / ab2,
        @as(T, 0.0),
        @as(T, 1.0),
    );
    const closest = seg.a.displace(t * abx, t * aby, t * abz);
    return if (dim == 3)
        point.distance(T, p, closest)
    else
        point.distance_2d(T, p, closest);
}

/// Shortest 3D distance from point `p` to segment `seg`.
pub fn distance_point_to_segment(comptime T: type, seg: Segment(T), p: point.Point(T)) T {
    return distance_point_to_segment_dim(T, 3, seg, p);
}

/// Shortest 2D distance from point `p` to segment `seg` (ignoring z).
pub fn distance_point_to_segment_2d(comptime T: type, seg: Segment(T), p: point.Point(T)) T {
    return distance_point_to_segment_dim(T, 2, seg, p);
}

test "Segment constructors" {
    const S = Segment(f64);
    const a = S.new_2d(0.0, 0.0, 1.0, 1.0);

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), a.a.x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), a.b.x, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), a.a.z, 1e-12);
}

test "distance from point to segment" {
    const P = point.Point(f64);
    const S = Segment(f64);

    const seg = S.new_2d(0.0, 0.0, 4.0, 0.0);

    try std.testing.expectApproxEqAbs(@as(f64, 3.0), distance_point_to_segment_2d(f64, seg, P.new_2d(1.0, -3.0)), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), distance_point_to_segment_2d(f64, seg, P.new_2d(-3.0, 4.0)), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), distance_point_to_segment_2d(f64, seg, P.new_2d(7.0, 4.0)), 1e-12);

    const seg3 = S.new(P.new(0.0, 0.0, 0.0), P.new(2.0, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), distance_point_to_segment(f64, seg3, P.new(1.0, 1.0, 0.0)), 1e-12);

    // Zero-length 3D segment falls back to point-to-point distance.
    const zero_seg = S.new(P.new(1.0, 2.0, 2.0), P.new(1.0, 2.0, 2.0));
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), distance_point_to_segment(f64, zero_seg, P.new(1.0, 2.0, 5.0)), 1e-12);

    // 3D endpoint projection: point projects beyond endpoint b.
    const seg_end = S.new(P.new(0.0, 0.0, 0.0), P.new(1.0, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f64, @sqrt(18.0)), distance_point_to_segment(f64, seg_end, P.new(4.0, 3.0, 0.0)), 1e-12);
}
