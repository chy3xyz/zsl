const std = @import("std");

/// Errors returned by the OpenCL/VCL stub backend.
pub const VclError = error{
    OutOfMemory,
    NotImplemented,
    InvalidDimension,
};

/// OpenCL device type bitmask, matching the VSL `vcl.DeviceType` enum.
pub const DeviceType = enum(i64) {
    default = 1 << 0,
    cpu = 1 << 1,
    gpu = 1 << 2,
    accelerator = 1 << 3,
    custom = 1 << 4,
    all = 0xFFFFFFFF,
};

/// Memory flag bitmask, matching the VCL `mem_*` constants.
pub const MemFlags = packed struct(u64) {
    read_write: bool = false,
    write_only: bool = false,
    read_only: bool = false,
    use_host_ptr: bool = false,
    alloc_host_ptr: bool = false,
    copy_host_ptr: bool = false,
    _reserved0: u1 = 0,
    host_write_only: bool = false,
    host_read_only: bool = false,
    host_no_access: bool = false,
    svm_fine_grain_buffer: bool = false,
    svm_atomics: bool = false,
    kernel_read_and_write: bool = false,
    _reserved1: u52 = 0,
};

test "VCL device type constants match VSL" {
    try std.testing.expectEqual(@as(i64, 1 << 0), @intFromEnum(DeviceType.default));
    try std.testing.expectEqual(@as(i64, 1 << 1), @intFromEnum(DeviceType.cpu));
    try std.testing.expectEqual(@as(i64, 1 << 2), @intFromEnum(DeviceType.gpu));
    try std.testing.expectEqual(@as(i64, 1 << 3), @intFromEnum(DeviceType.accelerator));
    try std.testing.expectEqual(@as(i64, 1 << 4), @intFromEnum(DeviceType.custom));
    try std.testing.expectEqual(@as(i64, 0xFFFFFFFF), @intFromEnum(DeviceType.all));
}
