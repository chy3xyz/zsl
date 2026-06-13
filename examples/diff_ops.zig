const std = @import("std");
const zsl = @import("zsl");

fn square(x: f64) f64 {
    return x * x;
}

fn sine(x: f64) f64 {
    return std.math.sin(x);
}

pub fn main() !void {
    const x: f64 = 3.0;

    const r_backward = zsl.diff.backward(square, x);
    std.debug.print("backward(x^2, {d}) value = {d}, err = {d}\n", .{ x, r_backward.value, r_backward.err });

    const r_forward = zsl.diff.forward(square, x);
    std.debug.print("forward(x^2, {d})  value = {d}, err = {d}\n", .{ x, r_forward.value, r_forward.err });

    const r_central = zsl.diff.central(square, x);
    std.debug.print("central(x^2, {d})  value = {d}, err = {d}\n", .{ x, r_central.value, r_central.err });

    const r_sin = zsl.diff.central(sine, 0.0);
    std.debug.print("central(sin, 0)    value = {d}, err = {d}\n", .{ r_sin.value, r_sin.err });
}
