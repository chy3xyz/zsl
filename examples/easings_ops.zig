const std = @import("std");
const zsl = @import("zsl");
const easings = zsl.easings;

pub fn main() !void {
    const t: f64 = 0.5;

    std.debug.print("Easing samples at t = {d}\n", .{t});
    std.debug.print("  linear:              {d}\n", .{easings.linear(t)});
    std.debug.print("  ease_in_quad:        {d}\n", .{easings.ease_in_quad(t)});
    std.debug.print("  ease_out_quad:       {d}\n", .{easings.ease_out_quad(t)});
    std.debug.print("  ease_in_out_quad:    {d}\n", .{easings.ease_in_out_quad(t)});
    std.debug.print("  ease_in_sine:        {d}\n", .{easings.ease_in_sine(t)});
    std.debug.print("  ease_out_circ:       {d}\n", .{easings.ease_out_circ(t)});
    std.debug.print("  ease_in_back:        {d}\n", .{easings.ease_in_back(t)});
    std.debug.print("  ease_in_back_overshoot(2): {d}\n", .{easings.ease_in_back_overshoot(t, 2.0)});
    std.debug.print("  ease_out_bounce:     {d}\n", .{easings.ease_out_bounce(t)});
    std.debug.print("  ease_in_elastic:     {d}\n", .{easings.ease_in_elastic(t)});
    std.debug.print("  ease_in_elastic_overshoot(2): {d}\n", .{easings.ease_in_elastic_overshoot(t, 2.0)});
}
