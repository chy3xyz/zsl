const std = @import("std");
const zsl = @import("zsl");
const noise = zsl.noise;

pub fn main() !void {
    const seed = 42;
    const g = noise.Generator.init(seed);

    std.debug.print("Noise demo (seed = {d})\n", .{seed});
    std.debug.print("perlin2d(0.5, 0.5)   = {d}\n", .{noise.perlin2d(g, 0.5, 0.5)});
    std.debug.print("perlin3d(0.5, 0.5, 0.5) = {d}\n", .{noise.perlin3d(g, 0.5, 0.5, 0.5)});
    std.debug.print("simplex2d(0.5, 0.5)  = {d}\n", .{noise.simplex2d(g, 0.5, 0.5)});
    std.debug.print("simplex3d(0.5, 0.5, 0.5) = {d}\n", .{noise.simplex3d(g, 0.5, 0.5, 0.5)});

    const default_g = noise.Generator.init_default();
    std.debug.print("Default generator perlin2d(1.0, 2.0) = {d}\n", .{noise.perlin2d(default_g, 1.0, 2.0)});
}
