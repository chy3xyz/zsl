pub const Error = error{
    OutOfMemory,
    InvalidDimension,
    ShapeMismatch,
    IndexOutOfBounds,
    DivisionByZero,
    NotImplemented,
    SingularMatrix,
    NotPositiveDefinite,
    NotConverged,
    NotFitted,
    FileReadError,
    FileWriteError,
    ParseError,
};

test "Error contains expected variants" {
    const e: Error = error.ShapeMismatch;
    try @import("std").testing.expect(e == error.ShapeMismatch);
}
