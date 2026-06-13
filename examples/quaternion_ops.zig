const std = @import("std");
const zsl = @import("zsl");
const quaternion = zsl.quaternion;

pub fn main() !void {
    std.debug.print("== quaternion operations ==\n", .{});

    // Identity quaternion
    const identity = quaternion.id();
    std.debug.print("identity: {}\n", .{identity});
    std.debug.print("is_identity = {}\n", .{identity.is_identity()});

    // Axis-angle creation: 90 degrees around Z axis
    const q_z = quaternion.from_axis_angle(std.math.pi / 2.0, 0.0, 0.0, 1.0);
    std.debug.print("90 deg around Z: {}\n", .{q_z});
    std.debug.print("norm = {d}\n", .{q_z.norm()});

    // Multiplication composes rotations
    const q_x = quaternion.from_axis_angle(std.math.pi / 4.0, 1.0, 0.0, 0.0);
    const combined = q_x.mul(q_z);
    std.debug.print("combined rotation: {}\n", .{combined});
    std.debug.print("combined is unit = {}\n", .{combined.abs() - 1.0 < 1e-12});

    // Rotate a vector
    const v = [3]f64{ 1.0, 0.0, 0.0 };
    const rotated = q_z.rotate_vector(v);
    std.debug.print("rotate (1,0,0) by 90 deg Z = ({d:.12}, {d:.12}, {d:.12})\n", .{ rotated[0], rotated[1], rotated[2] });
}
