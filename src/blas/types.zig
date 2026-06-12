pub const Transpose = enum {
    no_trans,
    trans,
    conj_trans,
};

test "Transpose variants exist" {
    const t: Transpose = .trans;
    try @import("std").testing.expect(t == .trans);
}
