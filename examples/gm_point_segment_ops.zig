const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const P = zsl.gm.Point(f64);
    const S = zsl.gm.Segment(f64);

    const a = P.new(1.0, 2.0, 3.0);
    const b = P.new_2d(4.0, 6.0);

    std.debug.print("Point a: ({d}, {d}, {d})\n", .{ a.x, a.y, a.z });
    std.debug.print("Point b: ({d}, {d}, {d})\n", .{ b.x, b.y, b.z });

    const d3 = zsl.gm.point.distance(f64, a, b);
    const d2 = zsl.gm.point.distance_2d(f64, a, b);
    const dsq = zsl.gm.point.distance_squared(f64, a, b);

    std.debug.print("3D distance(a, b)   = {d:.6}\n", .{d3});
    std.debug.print("2D distance(a, b)   = {d:.6}\n", .{d2});
    std.debug.print("Squared distance    = {d:.6}\n", .{dsq});

    const c = a.displace(2.0, -1.0, 0.5);
    std.debug.print("a displaced         = ({d}, {d}, {d})\n", .{ c.x, c.y, c.z });

    const seg = S.new_2d(0.0, 0.0, 10.0, 0.0);
    const p = P.new_2d(5.0, 4.0);
    const dist_to_seg = zsl.gm.segment.distance_point_to_segment_2d(f64, seg, p);
    std.debug.print("Distance from point to segment = {d:.6}\n", .{dist_to_seg});
}
