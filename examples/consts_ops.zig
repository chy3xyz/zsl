const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    std.debug.print("speed of light (MKS) = {e}\n", .{zsl.consts.mks.mks_speed_of_light});
    std.debug.print("Avogadro's number = {e}\n", .{zsl.consts.num.num_avogadro});
    std.debug.print("kilo prefix = {d}\n", .{zsl.consts.num.num_kilo});
}
