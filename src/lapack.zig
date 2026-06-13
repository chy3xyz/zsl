pub const lu = @import("lapack/lu.zig");
pub const helpers = @import("lapack/helpers.zig");
pub const qr = @import("lapack/qr.zig");
pub const cholesky = @import("lapack/cholesky.zig");
pub const eigen = @import("lapack/eigen.zig");
pub const svd = @import("lapack/svd.zig");

test {
    _ = lu;
    _ = helpers;
    _ = qr;
    _ = cholesky;
    _ = eigen;
    _ = svd;
}
