pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans,
};

pub const Uplo = enum {
    upper,
    lower,
};

pub const Side = enum {
    left,
    right,
};

pub const Diagonal = enum {
    unit,
    non_unit,
};

test "Transpose variants exist" {
    const t: Transpose = .trans;
    try @import("std").testing.expect(t == .trans);
}

test "BLAS control enums" {
    const u: Uplo = .upper;
    const s: Side = .left;
    const d: Diagonal = .unit;
    try @import("std").testing.expect(u == .upper);
    try @import("std").testing.expect(s == .left);
    try @import("std").testing.expect(d == .unit);
}
