const std = @import("std");
const blas = @import("../blas.zig");
const la = @import("../la.zig");
const util = @import("../util.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;
const Transpose = blas.Transpose;

test "lapack placeholder" {
    try std.testing.expect(true);
}
