const std = @import("std");

/// Errors returned by the Vulkan stub backend.
pub const VulkanError = error{
    OutOfMemory,
    NotImplemented,
    InvalidDimension,
};

/// Hint describing how a buffer will be used.
pub const BufferUsage = enum {
    vertex,
    index,
    uniform,
    storage,
    transfer_src,
    transfer_dst,
};
