const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;
    const D = zsl.ml.data.Data(f64);
    const LinReg = zsl.ml.linreg.LinReg;

    // y = 2*x + 1
    var data = try D.fromRawXy(allocator, &[_][]const f64{
        &[_]f64{0.0},
        &[_]f64{1.0},
        &[_]f64{2.0},
        &[_]f64{3.0},
        &[_]f64{4.0},
    }, &[_]f64{ 1.0, 3.0, 5.0, 7.0, 9.0 });
    defer data.deinit(allocator);

    var reg = try LinReg.init(&data, "demo", allocator);
    defer reg.deinit(allocator);

    try reg.train(allocator);

    std.debug.print("theta = {d}\n", .{try reg.params.get_theta(0)});
    std.debug.print("bias = {d}\n", .{reg.params.get_bias()});
    std.debug.print("predict(5) = {d}\n", .{reg.predict(&[_]f64{5.0})});
    std.debug.print("cost = {d}\n", .{reg.cost()});
}
