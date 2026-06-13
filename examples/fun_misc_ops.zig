const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    std.debug.print("=== fun misc: interpolation and sinusoid ===\n", .{});

    // Linear interpolation (already in zsl.fun.interp).
    const x = &[_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const y = &[_]f64{ 2.0, 5.0, 7.0, 10.0 };
    std.debug.print("linear interp at 1.5 = {d}\n", .{try zsl.fun.interp.data_interp(1.5, x, y, allocator)});
    std.debug.print("linear interp at 2.5 = {d}\n", .{try zsl.fun.interp.data_interp(2.5, x, y, allocator)});

    // Cubic interpolation via InterpCubic.
    var ic = zsl.fun.interp.InterpCubic.init();
    try ic.fit_4points(1.0, 2.0, 2.0, 5.0, 3.0, 7.0, 4.0, 10.0);
    std.debug.print("cubic interp at 1.5  = {d}\n", .{ic.f(1.5)});
    std.debug.print("cubic interp at 2.5  = {d}\n", .{ic.f(2.5)});

    // Quadratic interpolation via InterpQuad.
    var iq = zsl.fun.interp.InterpQuad.init();
    try iq.fit_3points(1.0, 2.0, 2.0, 5.0, 3.0, 7.0);
    std.debug.print("quad interp at 1.5   = {d}\n", .{iq.f(1.5)});
    std.debug.print("quad interp at 2.5   = {d}\n", .{iq.f(2.5)});

    // Sinusoid evaluation.
    const s = zsl.fun.sinusoid.Sinusoid.init(2.0, 1.0, 0.0, 1.0); // A=2, f=1, phase=0, offset=1
    std.debug.print("sinusoid at t=0.00   = {d}\n", .{s.evaluate(0.0)});
    std.debug.print("sinusoid at t=0.25   = {d}\n", .{s.evaluate(0.25)});
    std.debug.print("sinusoid at t=0.50   = {d}\n", .{s.evaluate(0.5)});
}
