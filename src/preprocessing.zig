pub const scalers = @import("preprocessing/scalers.zig");
pub const encoders = @import("preprocessing/encoders.zig");
pub const binning = @import("preprocessing/binning.zig");

// Convenience re-exports for new preprocessing extensions.
pub const RobustScaler = scalers.RobustScaler;
pub const digitize = binning.digitize;

test {
    _ = scalers;
    _ = encoders;
    _ = binning;
}
