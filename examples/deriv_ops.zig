const std = @import("std");
const zsl = @import("zsl");

fn square(x: f64) f64 {
    return x * x;
}

fn sine(x: f64) f64 {
    return std.math.sin(x);
}

fn product(v: []const f64) f64 {
    return v[0] * v[1];
}

pub fn main() !void {
    const h = 1.0e-4;

    const central_square = zsl.deriv.central(square, 3.0, h);
    std.debug.print("central d/dx x^2 at x=3: value={d}, err={d}\n", .{ central_square.value, central_square.err });

    const forward_square = zsl.deriv.forward(square, 3.0, h);
    std.debug.print("forward d/dx x^2 at x=3: value={d}, err={d}\n", .{ forward_square.value, forward_square.err });

    const central_sine = zsl.deriv.central(sine, 0.0, h);
    std.debug.print("central d/dx sin(x) at x=0: value={d}, err={d}\n", .{ central_sine.value, central_sine.err });

    const point = [_]f64{ 2.0, 3.0 };
    const partial_x = try zsl.deriv.partial(product, &point, 0, h);
    std.debug.print("partial d/dx x*y at (2,3): value={d}, err={d}\n", .{ partial_x.value, partial_x.err });
}
