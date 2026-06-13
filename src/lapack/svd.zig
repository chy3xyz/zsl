const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;
const Vector = la.Vector;

const SvdResult = struct {
    u: Matrix(f64),
    s: []f64,
    vt: Matrix(f64),
};

/// Core one-sided Jacobi SVD assuming m >= n.
///
/// Note: this implementation forms B = A^T * A and then diagonalizes B. This
/// squares the condition number of A and is therefore less numerically stable
/// than a true one-sided Jacobi method; it is acceptable for an initial,
/// small-dense-matrix implementation.
fn svd_m_geq_n(allocator: std.mem.Allocator, a: Matrix(f64)) Error!SvdResult {
    const m = a.rows;
    const n = a.cols;
    std.debug.assert(m >= n);

    const blas = @import("../blas.zig");
    const float = @import("../float.zig");

    // B = A^T * A (n×n symmetric).
    var b = try Matrix(f64).init(allocator, n, n);
    defer b.deinit(allocator);
    try blas.gemm(f64, .trans, .no_trans, 1.0, a, a, 0.0, &b);

    // Enforce symmetry in the face of round-off.
    for (0..n) |i| {
        for (i + 1..n) |j| {
            const avg = 0.5 * (try b.get(i, j) + try b.get(j, i));
            try b.set(i, j, avg);
            try b.set(j, i, avg);
        }
    }

    // Diagonalize B via Jacobi: eigenvalues -> v, eigenvectors -> q (columns).
    var q = try Matrix(f64).init(allocator, n, n);
    errdefer q.deinit(allocator);
    var v = try Vector(f64).init(allocator, n);
    errdefer v.deinit(allocator);

    try la.jacobi.jacobi(&q, &v, &b, allocator);

    // Singular values are square roots of eigenvalues.
    var s = try allocator.alloc(f64, n);
    errdefer allocator.free(s);
    for (0..n) |i| {
        const val = v.data[i * v.stride];
        s[i] = std.math.sqrt(@max(val, 0.0));
    }

    // Sort singular values (and corresponding eigenvectors) in descending order.
    var perm = try allocator.alloc(usize, n);
    defer allocator.free(perm);
    for (0..n) |i| {
        perm[i] = i;
    }
    for (1..n) |i| {
        var k = i;
        while (k > 0 and s[perm[k]] > s[perm[k - 1]]) {
            const tmp = perm[k];
            perm[k] = perm[k - 1];
            perm[k - 1] = tmp;
            k -= 1;
        }
    }

    var q_sorted = try Matrix(f64).init(allocator, n, n);
    errdefer q_sorted.deinit(allocator);
    var s_sorted = try allocator.alloc(f64, n);
    errdefer allocator.free(s_sorted);
    for (0..n) |j| {
        s_sorted[j] = s[perm[j]];
        for (0..n) |i| {
            try q_sorted.set(i, j, try q.get(i, perm[j]));
        }
    }

    q.deinit(allocator);
    allocator.free(s);
    v.deinit(allocator);

    // U = A * V.
    var u = try Matrix(f64).init(allocator, m, n);
    errdefer u.deinit(allocator);
    try blas.gemm(f64, .no_trans, .no_trans, 1.0, a, q_sorted, 0.0, &u);

    // Normalize columns of U by the singular values.
    const eps = float.eps(f64);
    const zero_tol = @max(eps, 1e-12);
    for (0..n) |j| {
        const sj = s_sorted[j];
        if (sj > zero_tol) {
            for (0..m) |i| {
                try u.set(i, j, try u.get(i, j) / sj);
            }
        }
    }

    // For rank-deficient matrices, columns of U corresponding to zero singular
    // values collapse to near zero. Replace each such column with a unit vector
    // orthogonal to all preceding columns of U (Gram-Schmidt against previous
    // columns using standard basis vectors as candidates).
    for (0..n) |j| {
        if (s_sorted[j] > zero_tol) continue;

        var candidate = try Vector(f64).init(allocator, m);
        defer candidate.deinit(allocator);

        var found = false;
        for (0..m) |basis_idx| {
            @memset(candidate.data, 0.0);
            candidate.data[basis_idx] = 1.0;

            for (0..j) |i| {
                const ui = try u.col(i);
                const proj = try blas.dot(f64, ui, candidate);
                for (0..m) |row| {
                    try candidate.set(row, try candidate.get(row) - proj * try ui.get(row));
                }
            }

            const norm = try blas.nrm2(f64, candidate);
            if (norm > zero_tol) {
                try blas.scal(f64, 1.0 / norm, &candidate);
                for (0..m) |row| {
                    try u.set(row, j, try candidate.get(row));
                }
                found = true;
                break;
            }
        }
        if (!found) {
            // All standard basis vectors lie in the span of previous columns;
            // this can only happen when m == j, in which case the column is
            // left as zero and the thin-SVD orthonormality contract is
            // preserved vacuously.
        }
    }

    // V^T = V^T.
    var vt = try Matrix(f64).init(allocator, n, n);
    errdefer vt.deinit(allocator);
    for (0..n) |i| {
        for (0..n) |j| {
            try vt.set(i, j, try q_sorted.get(j, i));
        }
    }

    q_sorted.deinit(allocator);

    return .{
        .u = u,
        .s = s_sorted,
        .vt = vt,
    };
}

/// One-sided Jacobi SVD of a dense matrix A.
///
/// Returns U, singular values s, and V^T such that A ≈ U * diag(s) * V^T.
/// For m >= n, U is m×n, s has length n, and V^T is n×n.
/// For m < n, the transpose trick is used and U is m×m, s has length m,
/// and V^T is m×n (thin SVD).
///
/// Note: the current implementation diagonalizes A^T * A, which squares the
/// condition number. This is acceptable for an initial small-dense-matrix
/// implementation but should be replaced by a true one-sided Jacobi method
/// for production use on ill-conditioned data.
pub fn dgesvd(allocator: std.mem.Allocator, a: Matrix(f64)) Error!SvdResult {
    if (a.rows >= a.cols) {
        return svd_m_geq_n(allocator, a);
    }

    // m < n: compute SVD of A^T and swap U/Vt.
    const at = a.transpose();
    var inner = try svd_m_geq_n(allocator, at);
    errdefer inner.u.deinit(allocator);
    errdefer allocator.free(inner.s);
    errdefer inner.vt.deinit(allocator);

    const k = inner.s.len; // = min(m, n)

    // U = V' = inner.vt^T.
    var u = try Matrix(f64).init(allocator, k, k);
    errdefer u.deinit(allocator);
    for (0..k) |i| {
        for (0..k) |j| {
            try u.set(i, j, try inner.vt.get(j, i));
        }
    }

    // Vt = U'^T = inner.u^T.
    var vt = try Matrix(f64).init(allocator, k, a.cols);
    errdefer vt.deinit(allocator);
    for (0..k) |i| {
        for (0..a.cols) |j| {
            try vt.set(i, j, try inner.u.get(j, i));
        }
    }

    const s = inner.s;
    inner.u.deinit(allocator);
    inner.vt.deinit(allocator);

    return .{
        .u = u,
        .s = s,
        .vt = vt,
    };
}

test "dgesvd tall matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 4, 3, &[_]T{
        1.0, 2.0,  3.0,
        4.0, 5.0,  6.0,
        7.0, 8.0,  10.0,
        9.0, 11.0, 12.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try dgesvd(std.testing.allocator, a);
    defer result.u.deinit(std.testing.allocator);
    defer std.testing.allocator.free(result.s);
    defer result.vt.deinit(std.testing.allocator);

    const k = result.s.len;
    try std.testing.expectEqual(@as(usize, 3), k);

    // U^T * U ≈ I.
    var utu = try M.init(std.testing.allocator, k, k);
    defer utu.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.u, result.u, 0.0, &utu);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try utu.get(i, j), expected, 1e-9));
        }
    }

    // V^T * V ≈ I.
    var vtv = try M.init(std.testing.allocator, k, k);
    defer vtv.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, result.vt, result.vt, 0.0, &vtv);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try vtv.get(i, j), expected, 1e-9));
        }
    }

    // U * diag(s) * V^T ≈ A.
    var s_mat = try M.init(std.testing.allocator, k, k);
    defer s_mat.deinit(std.testing.allocator);
    for (0..k) |i| {
        try s_mat.set(i, i, result.s[i]);
    }

    var us = try M.init(std.testing.allocator, a.rows, k);
    defer us.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.u, s_mat, 0.0, &us);

    var reconstructed = try M.init(std.testing.allocator, a.rows, a.cols);
    defer reconstructed.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, us, result.vt, 0.0, &reconstructed);

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expect(float.approxEqAbs(T, try reconstructed.get(i, j), try a.get(i, j), 1e-8));
        }
    }
}

test "dgesvd square matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 3, 3, &[_]T{
        3.0, 1.0, 2.0,
        1.0, 4.0, 1.0,
        2.0, 1.0, 5.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try dgesvd(std.testing.allocator, a);
    defer result.u.deinit(std.testing.allocator);
    defer std.testing.allocator.free(result.s);
    defer result.vt.deinit(std.testing.allocator);

    const k = result.s.len;

    var utu = try M.init(std.testing.allocator, k, k);
    defer utu.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.u, result.u, 0.0, &utu);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try utu.get(i, j), expected, 1e-9));
        }
    }

    var vtv = try M.init(std.testing.allocator, k, k);
    defer vtv.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, result.vt, result.vt, 0.0, &vtv);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try vtv.get(i, j), expected, 1e-9));
        }
    }

    var s_mat = try M.init(std.testing.allocator, k, k);
    defer s_mat.deinit(std.testing.allocator);
    for (0..k) |i| {
        try s_mat.set(i, i, result.s[i]);
    }

    var us = try M.init(std.testing.allocator, a.rows, k);
    defer us.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.u, s_mat, 0.0, &us);

    var reconstructed = try M.init(std.testing.allocator, a.rows, a.cols);
    defer reconstructed.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, us, result.vt, 0.0, &reconstructed);

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expect(float.approxEqAbs(T, try reconstructed.get(i, j), try a.get(i, j), 1e-8));
        }
    }
}

test "dgesvd wide matrix via transpose" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    var a = try M.fromRowSlice(std.testing.allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try dgesvd(std.testing.allocator, a);
    defer result.u.deinit(std.testing.allocator);
    defer std.testing.allocator.free(result.s);
    defer result.vt.deinit(std.testing.allocator);

    const k = result.s.len;
    try std.testing.expectEqual(@as(usize, 2), k);

    var utu = try M.init(std.testing.allocator, k, k);
    defer utu.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.u, result.u, 0.0, &utu);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try utu.get(i, j), expected, 1e-9));
        }
    }

    var vtv = try M.init(std.testing.allocator, k, k);
    defer vtv.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, result.vt, result.vt, 0.0, &vtv);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try vtv.get(i, j), expected, 1e-9));
        }
    }

    var s_mat = try M.init(std.testing.allocator, k, k);
    defer s_mat.deinit(std.testing.allocator);
    for (0..k) |i| {
        try s_mat.set(i, i, result.s[i]);
    }

    var us = try M.init(std.testing.allocator, a.rows, k);
    defer us.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.u, s_mat, 0.0, &us);

    var reconstructed = try M.init(std.testing.allocator, a.rows, a.cols);
    defer reconstructed.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, us, result.vt, 0.0, &reconstructed);

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expect(float.approxEqAbs(T, try reconstructed.get(i, j), try a.get(i, j), 1e-8));
        }
    }
}

test "dgesvd handles rank-deficient matrix" {
    const T = f64;
    const M = Matrix(T);
    const float = @import("../float.zig");
    const blas = @import("../blas.zig");

    // Second column is twice the first column; rank is 1.
    var a = try M.fromRowSlice(std.testing.allocator, 3, 2, &[_]T{
        1.0, 2.0,
        2.0, 4.0,
        3.0, 6.0,
    });
    defer a.deinit(std.testing.allocator);

    var result = try dgesvd(std.testing.allocator, a);
    defer result.u.deinit(std.testing.allocator);
    defer std.testing.allocator.free(result.s);
    defer result.vt.deinit(std.testing.allocator);

    const k = result.s.len;
    try std.testing.expectEqual(@as(usize, 2), k);

    // The smaller singular value must be (numerically) zero.
    try std.testing.expect(float.approxEqAbs(T, result.s[1], 0.0, 1e-9));

    // U^T * U ≈ I.
    var utu = try M.init(std.testing.allocator, k, k);
    defer utu.deinit(std.testing.allocator);
    try blas.gemm(T, .trans, .no_trans, 1.0, result.u, result.u, 0.0, &utu);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try utu.get(i, j), expected, 1e-9));
        }
    }

    // V^T * V ≈ I.
    var vtv = try M.init(std.testing.allocator, k, k);
    defer vtv.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .trans, 1.0, result.vt, result.vt, 0.0, &vtv);
    for (0..k) |i| {
        for (0..k) |j| {
            const expected = if (i == j) @as(T, 1.0) else @as(T, 0.0);
            try std.testing.expect(float.approxEqAbs(T, try vtv.get(i, j), expected, 1e-9));
        }
    }

    // U * diag(s) * V^T ≈ A.
    var s_mat = try M.init(std.testing.allocator, k, k);
    defer s_mat.deinit(std.testing.allocator);
    for (0..k) |i| {
        try s_mat.set(i, i, result.s[i]);
    }

    var us = try M.init(std.testing.allocator, a.rows, k);
    defer us.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, result.u, s_mat, 0.0, &us);

    var reconstructed = try M.init(std.testing.allocator, a.rows, a.cols);
    defer reconstructed.deinit(std.testing.allocator);
    try blas.gemm(T, .no_trans, .no_trans, 1.0, us, result.vt, 0.0, &reconstructed);

    for (0..a.rows) |i| {
        for (0..a.cols) |j| {
            try std.testing.expect(float.approxEqAbs(T, try reconstructed.get(i, j), try a.get(i, j), 1e-8));
        }
    }
}
