const std = @import("std");
const Complex = @import("../fun/complex.zig").Complex;
const Error = @import("../errors.zig").Error;

const C64 = Complex(f64);

/// Return true when both real and imaginary parts are within `tol` of the expected value.
fn approxEqC64(actual: C64, expected: C64, tol: f64) bool {
    return @abs(actual.re - expected.re) <= tol and @abs(actual.im - expected.im) <= tol;
}

/// Complex SAXPY: y <- alpha * x + y
pub fn caxpy(alpha: C64, x: []const C64, y: []C64) Error!void {
    if (x.len != y.len) return error.ShapeMismatch;
    for (0..y.len) |i| {
        y[i] = alpha.mul(x[i]).add(y[i]);
    }
}

/// Unconjugated complex dot product: sum_i x[i] * y[i]
pub fn cdotu(x: []const C64, y: []const C64) Error!C64 {
    if (x.len != y.len) return error.ShapeMismatch;
    var sum = C64.new(0, 0);
    for (0..x.len) |i| {
        sum = sum.add(x[i].mul(y[i]));
    }
    return sum;
}

/// Conjugated complex dot product: sum_i conj(x[i]) * y[i]
pub fn cdotc(x: []const C64, y: []const C64) Error!C64 {
    if (x.len != y.len) return error.ShapeMismatch;
    var sum = C64.new(0, 0);
    for (0..x.len) |i| {
        sum = sum.add(x[i].conj().mul(y[i]));
    }
    return sum;
}

/// Complex general matrix-vector multiply, row-major layout.
/// A is m x n with leading dimension lda (lda >= n).
/// When trans == false: y <- alpha * A * x + beta * y.
/// When trans == true:  y <- alpha * A^T * x + beta * y.
pub fn cgemv(
    alpha: C64,
    A: []const C64,
    lda: usize,
    x: []const C64,
    beta: C64,
    y: []C64,
    m: usize,
    n: usize,
    trans: bool,
) Error!void {
    if (lda < n) return error.ShapeMismatch;
    if (A.len < lda * m) return error.ShapeMismatch;

    if (trans) {
        if (x.len < m) return error.ShapeMismatch;
        if (y.len < n) return error.ShapeMismatch;
    } else {
        if (x.len < n) return error.ShapeMismatch;
        if (y.len < m) return error.ShapeMismatch;
    }

    // y <- beta * y
    if (beta.re == 0 and beta.im == 0) {
        const dim = if (trans) n else m;
        for (0..dim) |i| y[i] = C64.new(0, 0);
    } else if (!(beta.re == 1 and beta.im == 0)) {
        const dim = if (trans) n else m;
        for (0..dim) |i| y[i] = beta.mul(y[i]);
    }

    if (alpha.re == 0 and alpha.im == 0) return;

    if (trans) {
        for (0..n) |j| {
            var sum = C64.new(0, 0);
            for (0..m) |i| {
                sum = sum.add(A[i * lda + j].mul(x[i]));
            }
            y[j] = y[j].add(alpha.mul(sum));
        }
    } else {
        for (0..m) |i| {
            var sum = C64.new(0, 0);
            for (0..n) |j| {
                sum = sum.add(A[i * lda + j].mul(x[j]));
            }
            y[i] = y[i].add(alpha.mul(sum));
        }
    }
}

/// Complex general matrix-matrix multiply, row-major layout, no-transpose case.
/// C <- alpha * A * B + beta * C
/// A is m x k (lda >= k), B is k x n (ldb >= n), C is m x n (ldc >= n).
pub fn cgemm(
    alpha: C64,
    A: []const C64,
    lda: usize,
    B: []const C64,
    ldb: usize,
    beta: C64,
    C: []C64,
    ldc: usize,
    m: usize,
    n: usize,
    k: usize,
) Error!void {
    if (lda < k) return error.ShapeMismatch;
    if (ldb < n) return error.ShapeMismatch;
    if (ldc < n) return error.ShapeMismatch;
    if (A.len < lda * m) return error.ShapeMismatch;
    if (B.len < ldb * k) return error.ShapeMismatch;
    if (C.len < ldc * m) return error.ShapeMismatch;

    // C <- beta * C
    if (beta.re == 0 and beta.im == 0) {
        for (0..m) |i| {
            for (0..n) |j| {
                C[i * ldc + j] = C64.new(0, 0);
            }
        }
    } else if (!(beta.re == 1 and beta.im == 0)) {
        for (0..m) |i| {
            for (0..n) |j| {
                C[i * ldc + j] = beta.mul(C[i * ldc + j]);
            }
        }
    }

    if (alpha.re == 0 and alpha.im == 0) return;

    for (0..m) |i| {
        for (0..n) |j| {
            var sum = C64.new(0, 0);
            for (0..k) |l| {
                sum = sum.add(A[i * lda + l].mul(B[l * ldb + j]));
            }
            C[i * ldc + j] = C[i * ldc + j].add(alpha.mul(sum));
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "caxpy" {
    var y = [_]C64{
        C64.new(1, 1),
        C64.new(2, -1),
        C64.new(0, 0),
    };
    const x = [_]C64{
        C64.new(1, 0),
        C64.new(0, 1),
        C64.new(2, 2),
    };
    const alpha = C64.new(2, 1);
    try caxpy(alpha, &x, &y);

    // y[0] = (2+i)*(1+0i) + (1+i) = 2+i+1+i = 3+2i
    try std.testing.expect(approxEqC64(y[0], C64.new(3, 2), 1e-12));
    // y[1] = (2+i)*(0+i) + (2-i) = (2i-1) + (2-i) = 1+i
    try std.testing.expect(approxEqC64(y[1], C64.new(1, 1), 1e-12));
    // y[2] = (2+i)*(2+2i) + 0 = 4+4i+2i-2 = 2+6i
    try std.testing.expect(approxEqC64(y[2], C64.new(2, 6), 1e-12));
}

test "cdotu" {
    const x = [_]C64{ C64.new(1, 2), C64.new(3, -1) };
    const y = [_]C64{ C64.new(2, 0), C64.new(1, 1) };
    const result = try cdotu(&x, &y);
    // x[0]*y[0] = (1+2i)*2 = 2+4i
    // x[1]*y[1] = (3-i)*(1+i) = 3+3i-i+1 = 4+2i
    // sum = 6+6i
    try std.testing.expect(approxEqC64(result, C64.new(6, 6), 1e-12));
}

test "cdotc" {
    const x = [_]C64{ C64.new(1, 2), C64.new(3, -1) };
    const y = [_]C64{ C64.new(2, 0), C64.new(1, 1) };
    const result = try cdotc(&x, &y);
    // conj(x[0])*y[0] = (1-2i)*2 = 2-4i
    // conj(x[1])*y[1] = (3+i)*(1+i) = 3+3i+i-1 = 2+4i
    // sum = 4+0i
    try std.testing.expect(approxEqC64(result, C64.new(4, 0), 1e-12));
}

test "cgemv no-trans" {
    // A = [[1+1i, 2+0i], [0-1i, 3+2i]]  (m=2, n=2)
    var A = [_]C64{
        C64.new(1, 1),  C64.new(2, 0),
        C64.new(0, -1), C64.new(3, 2),
    };
    const x = [_]C64{ C64.new(1, 1), C64.new(2, -1) };
    var y = [_]C64{ C64.new(0, 0), C64.new(0, 0) };
    const alpha = C64.new(1, 0);
    const beta = C64.new(0, 0);

    try cgemv(alpha, &A, 2, &x, beta, &y, 2, 2, false);

    // A*x = [ (1+i)(1+i) + 2(2-i), (-i)(1+i) + (3+2i)(2-i) ]
    //     = [ (1+i)^2 + 4 - 2i, -i(1+i) + (3+2i)(2-i) ]
    //     = [ 2i + 4 - 2i, -i + 1 + 6 - 3i + 4i + 2 ]
    //     = [ 4, 9 ]
    try std.testing.expect(approxEqC64(y[0], C64.new(4, 0), 1e-12));
    try std.testing.expect(approxEqC64(y[1], C64.new(9, 0), 1e-12));
}

test "cgemv trans" {
    // A = [[1+1i, 2+0i], [0-1i, 3+2i]]  (m=2, n=2)
    var A = [_]C64{
        C64.new(1, 1),  C64.new(2, 0),
        C64.new(0, -1), C64.new(3, 2),
    };
    const x = [_]C64{ C64.new(1, 0), C64.new(0, 1) };
    var y = [_]C64{ C64.new(0, 0), C64.new(0, 0) };

    try cgemv(C64.new(1, 0), &A, 2, &x, C64.new(0, 0), &y, 2, 2, true);

    // A^T * x = [ (1+i)*1 + (-i)*i, 2*1 + (3+2i)*i ]
    //         = [ 1+i + 1, 2 + 3i - 2 ]
    //         = [ 2+i, 3i ]
    try std.testing.expect(approxEqC64(y[0], C64.new(2, 1), 1e-12));
    try std.testing.expect(approxEqC64(y[1], C64.new(0, 3), 1e-12));
}

test "cgemv beta accumulation" {
    var A = [_]C64{
        C64.new(1, 0), C64.new(0, 1),
        C64.new(0, 1), C64.new(1, 0),
    };
    const x = [_]C64{ C64.new(1, 0), C64.new(0, 0) };
    var y = [_]C64{ C64.new(2, 1), C64.new(1, -1) };

    // A*x = [1, i]; y <- 1*A*x + 2*y = [1 + 4 + 2i, i + 2 - 2i] = [5+2i, 2-i]
    try cgemv(C64.new(1, 0), &A, 2, &x, C64.new(2, 0), &y, 2, 2, false);

    try std.testing.expect(approxEqC64(y[0], C64.new(5, 2), 1e-12));
    try std.testing.expect(approxEqC64(y[1], C64.new(2, -1), 1e-12));
}

test "cgemm no-trans" {
    // A = [[1+i, 2], [3, 1-i]] (m=2, k=2)
    var A = [_]C64{
        C64.new(1, 1), C64.new(2, 0),
        C64.new(3, 0), C64.new(1, -1),
    };
    // B = [[1, i], [2+i, 0]] (k=2, n=2)
    var B = [_]C64{
        C64.new(1, 0), C64.new(0, 1),
        C64.new(2, 1), C64.new(0, 0),
    };
    var C = [_]C64{
        C64.new(0, 0), C64.new(0, 0),
        C64.new(0, 0), C64.new(0, 0),
    };

    try cgemm(C64.new(1, 0), &A, 2, &B, 2, C64.new(0, 0), &C, 2, 2, 2, 2);

    // C[0,0] = (1+i)*1 + 2*(2+i) = 1+i + 4+2i = 5+3i
    try std.testing.expect(approxEqC64(C[0], C64.new(5, 3), 1e-12));
    // C[0,1] = (1+i)*i + 2*0 = i - 1
    try std.testing.expect(approxEqC64(C[1], C64.new(-1, 1), 1e-12));
    // C[1,0] = 3*1 + (1-i)*(2+i) = 3 + 2+i-2i+1 = 3 + 3 - i = 6 - i
    try std.testing.expect(approxEqC64(C[2], C64.new(6, -1), 1e-12));
    // C[1,1] = 3*i + (1-i)*0 = 3i
    try std.testing.expect(approxEqC64(C[3], C64.new(0, 3), 1e-12));
}

test "cgemm beta accumulation" {
    var A = [_]C64{
        C64.new(1, 0), C64.new(0, 0),
        C64.new(0, 0), C64.new(1, 0),
    };
    var B = [_]C64{
        C64.new(2, 0), C64.new(0, 0),
        C64.new(0, 0), C64.new(2, 0),
    };
    var C = [_]C64{
        C64.new(1, 1), C64.new(0, 0),
        C64.new(0, 0), C64.new(1, -1),
    };

    // C <- 1*A*B + i*C = 2*I + i*diag(1+i, 1-i)
    // diag(0) = 2 + i*(1+i) = 2 + i - 1 = 1 + i
    // diag(1) = 2 + i*(1-i) = 2 + i + 1 = 3 + i
    try cgemm(C64.new(1, 0), &A, 2, &B, 2, C64.new(0, 1), &C, 2, 2, 2, 2);

    try std.testing.expect(approxEqC64(C[0], C64.new(1, 1), 1e-12));
    try std.testing.expect(approxEqC64(C[3], C64.new(3, 1), 1e-12));
}

test "cgemm with leading dimensions" {
    // A is 2x2 stored in a 3-column buffer (lda=3).
    var A = [_]C64{
        C64.new(1, 0), C64.new(2, 0), C64.new(99, 99),
        C64.new(3, 0), C64.new(4, 0), C64.new(99, 99),
    };
    // B is 2x2 stored in a 4-column buffer (ldb=4).
    var B = [_]C64{
        C64.new(1, 0), C64.new(0, 1), C64.new(99, 99), C64.new(99, 99),
        C64.new(2, 0), C64.new(1, 0), C64.new(99, 99), C64.new(99, 99),
    };
    // C is 2x2 stored in a 4-column buffer (ldc=4).
    var C = [_]C64{
        C64.new(0, 0), C64.new(0, 0), C64.new(99, 99), C64.new(99, 99),
        C64.new(0, 0), C64.new(0, 0), C64.new(99, 99), C64.new(99, 99),
    };

    try cgemm(C64.new(1, 0), &A, 3, &B, 4, C64.new(0, 0), &C, 4, 2, 2, 2);

    // A*B = [[1*1+2*2, 1*i+2*1], [3*1+4*2, 3*i+4*1]] = [[5, 2+i], [11, 4+3i]]
    try std.testing.expect(approxEqC64(C[0], C64.new(5, 0), 1e-12));
    try std.testing.expect(approxEqC64(C[1], C64.new(2, 1), 1e-12));
    try std.testing.expect(approxEqC64(C[4], C64.new(11, 0), 1e-12));
    try std.testing.expect(approxEqC64(C[5], C64.new(4, 3), 1e-12));
}

test "complex blas shape mismatch returns error" {
    const x = [_]C64{C64.new(1, 0)};
    var y = [_]C64{ C64.new(0, 0), C64.new(0, 0) };
    try std.testing.expectError(error.ShapeMismatch, caxpy(C64.new(1, 0), &x, &y));
    try std.testing.expectError(error.ShapeMismatch, cdotu(&x, &y));
    try std.testing.expectError(error.ShapeMismatch, cdotc(&x, &y));
}
