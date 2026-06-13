const std = @import("std");
const la = @import("../la.zig");
const blas = @import("../blas.zig");
const float = @import("../float.zig");
const Error = @import("../errors.zig").Error;
const Data = @import("data.zig").Data;
const Stat = @import("workspace.zig").Stat;

/// Kernel function used by the SVM.
pub const KernelType = enum {
    linear,
    polynomial,
    rbf,
};

/// Support Vector Machine classifier.
///
/// Trains a binary classifier with labels `-1.0` and `1.0` using a simplified
/// Sequential Minimal Optimization (SMO) algorithm.
pub const SVM = struct {
    name: []const u8,
    data: *Data(f64),
    stat: *Stat(f64),
    alpha: []f64,
    bias: f64,
    kernel_type: KernelType,
    degree: usize,
    trained: bool,
    sv_data: []f64,
    support_vectors: []const []const f64,
    support_vector_labels: []const f64,
    support_vector_alphas: []const f64,
    c: f64,
    gamma: f64,

    const Self = @This();

    const sv_threshold = 1e-5;
    const kkt_tol = 1e-10;
    const eta_eps = 1e-10;

    /// Allocates a new SVM bound to `data`.
    pub fn init(data: *Data(f64), name: []const u8, allocator: std.mem.Allocator) Error!Self {
        if (data.y.len == 0) return error.InvalidDimension;

        for (data.y) |yi| {
            if (!float.approxEqAbs(f64, @abs(yi), 1.0, 1e-9)) {
                return error.InvalidDimension;
            }
        }

        const stat = try allocator.create(Stat(f64));
        errdefer allocator.destroy(stat);
        stat.* = try Stat(f64).from_data(data.*, name, allocator);

        const alpha = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(alpha);
        @memset(alpha, 0.0);

        // Flat buffer that will hold a copy of each support vector.
        const sv_data = try allocator.alloc(f64, data.nb_samples * data.nb_features);
        errdefer allocator.free(sv_data);
        @memset(sv_data, 0.0);

        // Pre-allocate buffers for support vectors so that `train` does not
        // need an allocator. The buffers are over-allocated and re-sliced to
        // the exact support-vector count after training.
        const support_vectors = try allocator.alloc([]const f64, data.nb_samples);
        errdefer allocator.free(support_vectors);
        @memset(support_vectors, &[_]f64{});

        const support_vector_labels = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(support_vector_labels);
        @memset(support_vector_labels, 0.0);

        const support_vector_alphas = try allocator.alloc(f64, data.nb_samples);
        errdefer allocator.free(support_vector_alphas);
        @memset(support_vector_alphas, 0.0);

        return .{
            .name = name,
            .data = data,
            .stat = stat,
            .alpha = alpha,
            .bias = 0.0,
            .kernel_type = .linear,
            .degree = 3,
            .trained = false,
            .sv_data = sv_data,
            .support_vectors = support_vectors[0..0],
            .support_vector_labels = support_vector_labels[0..0],
            .support_vector_alphas = support_vector_alphas[0..0],
            .c = 1.0,
            .gamma = 1.0,
        };
    }

    /// Releases all memory owned by the model (but not the underlying `data`).
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.stat.deinit(allocator);
        allocator.destroy(self.stat);
        allocator.free(self.alpha);
        allocator.free(self.sv_data);

        allocator.free(self.support_vectors.ptr[0..self.data.nb_samples]);
        allocator.free(self.support_vector_labels.ptr[0..self.data.nb_samples]);
        allocator.free(self.support_vector_alphas.ptr[0..self.data.nb_samples]);

        self.support_vectors = &[_][]const f64{};
        self.support_vector_labels = &[_]f64{};
        self.support_vector_alphas = &[_]f64{};
    }

    /// Computes the kernel function K(a, b).
    fn kernel(self: *const Self, a: []const f64, b: []const f64) f64 {
        switch (self.kernel_type) {
            .linear => {
                const va = la.Vector(f64){ .data = @constCast(a), .len = a.len, .stride = 1 };
                const vb = la.Vector(f64){ .data = @constCast(b), .len = b.len, .stride = 1 };
                return blas.dot(f64, va, vb) catch unreachable;
            },
            .polynomial => {
                const va = la.Vector(f64){ .data = @constCast(a), .len = a.len, .stride = 1 };
                const vb = la.Vector(f64){ .data = @constCast(b), .len = b.len, .stride = 1 };
                const d = blas.dot(f64, va, vb) catch unreachable;
                const base = self.gamma * d;
                var result: f64 = 1.0;
                for (0..self.degree) |_| {
                    result *= base;
                }
                return result;
            },
            .rbf => {
                var norm_sq: f64 = 0.0;
                for (a, b) |ai, bi| {
                    const diff = ai - bi;
                    norm_sq += diff * diff;
                }
                return @exp(-self.gamma * norm_sq);
            },
        }
    }

    /// Returns the raw decision value for `x` using the current alpha values.
    /// Used during training before support vectors have been extracted.
    fn decisionRaw(self: *const Self, x: []const f64) f64 {
        var result = self.bias;
        for (0..self.data.nb_samples) |i| {
            if (self.alpha[i] > kkt_tol) {
                const xi_vec = self.data.x.row(i) catch unreachable;
                const xi = xi_vec.data[0..xi_vec.len];
                result += self.alpha[i] * self.data.y[i] * self.kernel(xi, x);
            }
        }
        return result;
    }

    /// Returns the raw decision value for `x` using the extracted support vectors.
    pub fn decision(self: *const Self, x: []const f64) f64 {
        var result = self.bias;
        for (self.support_vectors, self.support_vector_alphas, self.support_vector_labels) |sv, alpha, label| {
            result += alpha * label * self.kernel(sv, x);
        }
        return result;
    }

    /// Trains the SVM using simplified SMO.
    pub fn train(self: *Self, max_passes: usize, tol: f64) Error!void {
        if (self.data.nb_samples == 0) return;

        @memset(self.alpha, 0.0);
        self.bias = 0.0;
        self.trained = false;

        const n = self.data.nb_samples;
        const C = self.c;

        var pass: usize = 0;
        while (pass < max_passes) : (pass += 1) {
            var num_changed: usize = 0;

            for (0..n) |i| {
                const xi_vec = try self.data.x.row(i);
                const xi = xi_vec.data[0..xi_vec.len];
                const yi = self.data.y[i];
                const ei = self.decisionRaw(xi) - yi;

                if ((yi * ei < -tol and self.alpha[i] < C) or
                    (yi * ei > tol and self.alpha[i] > 0.0))
                {
                    var j = (i + 1) % n;
                    if (j == i) {
                        j = (j + 1) % n;
                    }
                    if (j == i) continue;

                    const xj_vec = try self.data.x.row(j);
                    const xj = xj_vec.data[0..xj_vec.len];
                    const yj = self.data.y[j];
                    const ej = self.decisionRaw(xj) - yj;

                    var L: f64 = 0.0;
                    var H: f64 = C;
                    if (yi == yj) {
                        L = @max(0.0, self.alpha[i] + self.alpha[j] - C);
                        H = @min(C, self.alpha[i] + self.alpha[j]);
                    } else {
                        L = @max(0.0, self.alpha[j] - self.alpha[i]);
                        H = @min(C, C + self.alpha[j] - self.alpha[i]);
                    }

                    if (@abs(L - H) < eta_eps) continue;

                    const kii = self.kernel(xi, xi);
                    const kij = self.kernel(xi, xj);
                    const kjj = self.kernel(xj, xj);
                    const eta = kii + kjj - 2.0 * kij;

                    if (eta <= 0.0) continue;

                    const alpha_j_old = self.alpha[j];
                    var alpha_j_new = alpha_j_old + yj * (ei - ej) / eta;
                    if (alpha_j_new > H) {
                        alpha_j_new = H;
                    } else if (alpha_j_new < L) {
                        alpha_j_new = L;
                    }

                    const alpha_i_old = self.alpha[i];
                    self.alpha[j] = alpha_j_new;
                    self.alpha[i] = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new);

                    const b1 = self.bias - ei -
                        yi * (self.alpha[i] - alpha_i_old) * kii -
                        yj * (self.alpha[j] - alpha_j_old) * kij;
                    const b2 = self.bias - ej -
                        yi * (self.alpha[i] - alpha_i_old) * kij -
                        yj * (self.alpha[j] - alpha_j_old) * kjj;

                    if (self.alpha[i] > 0.0 and self.alpha[i] < C) {
                        self.bias = b1;
                    } else if (self.alpha[j] > 0.0 and self.alpha[j] < C) {
                        self.bias = b2;
                    } else {
                        self.bias = (b1 + b2) / 2.0;
                    }

                    num_changed += 1;
                }
            }

            if (num_changed == 0) break;
        }

        try self.extractSupportVectors();
        self.trained = true;
    }

    /// Returns the predicted class label (`-1.0` or `1.0`) for `x`.
    pub fn predict(self: *const Self, x: []const f64) Error!f64 {
        if (!self.trained) return error.NotFitted;
        const raw = self.decision(x);
        return if (raw >= 0.0) 1.0 else -1.0;
    }

    /// Extracts the support vectors (alpha > threshold) into pre-allocated buffers.
    fn extractSupportVectors(self: *Self) Error!void {
        var count: usize = 0;
        var sv = @constCast(self.support_vectors.ptr[0..self.data.nb_samples]);
        var sv_labels = @constCast(self.support_vector_labels.ptr[0..self.data.nb_samples]);
        var sv_alphas = @constCast(self.support_vector_alphas.ptr[0..self.data.nb_samples]);
        for (0..self.data.nb_samples) |i| {
            if (self.alpha[i] > sv_threshold) {
                const xi_vec = try self.data.x.row(i);
                const xi = xi_vec.data[0..xi_vec.len];
                const offset = count * self.data.nb_features;
                const dst = self.sv_data[offset .. offset + self.data.nb_features];
                @memcpy(dst, xi);
                sv[count] = dst;
                sv_labels[count] = self.data.y[i];
                sv_alphas[count] = self.alpha[i];
                count += 1;
            }
        }
        self.support_vectors = self.support_vectors.ptr[0..count];
        self.support_vector_labels = self.support_vector_labels.ptr[0..count];
        self.support_vector_alphas = self.support_vector_alphas.ptr[0..count];
    }
};

test {
    const allocator = std.testing.allocator;
    const D = Data(f64);

    const xraw = &[_][]const f64{
        &[_]f64{ 0.0, 0.0 },
        &[_]f64{ 1.0, 0.0 },
        &[_]f64{ 0.0, 1.0 },
        &[_]f64{ 3.0, 3.0 },
        &[_]f64{ 4.0, 3.0 },
        &[_]f64{ 3.0, 4.0 },
    };
    const yraw = &[_]f64{ -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 };

    var data = try D.fromRawXy(allocator, xraw, yraw);
    defer data.deinit(allocator);

    var svm = try SVM.init(&data, "test_svm", allocator);
    defer svm.deinit(allocator);

    try std.testing.expectError(error.NotFitted, svm.predict(&[_]f64{ 0.0, 0.0 }));

    try svm.train(1000, 1e-3);

    try std.testing.expect(svm.trained);
    try std.testing.expect(svm.support_vectors.len > 0);

    const pred_neg = try svm.predict(&[_]f64{ -1.0, -1.0 });
    const pred_pos = try svm.predict(&[_]f64{ 5.0, 5.0 });

    try std.testing.expectApproxEqAbs(@as(f64, -1.0), pred_neg, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), pred_pos, 1e-9);
}
