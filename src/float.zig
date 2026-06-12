const std = @import("std");

pub fn eps(comptime T: type) T {
    return @as(T, 0);
}
