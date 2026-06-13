const std = @import("std");
const Error = @import("../errors.zig").Error;
const Matrix = @import("../la.zig").Matrix;

pub const RocCurveResult = struct {
    fpr: []f64,
    tpr: []f64,
    thresholds: []f64,
};

pub const PrecisionRecallResult = struct {
    precision: []f64,
    recall: []f64,
    thresholds: []f64,
};

fn checkLength(a: usize, b: usize) Error!void {
    if (a != b) return error.ShapeMismatch;
}

fn checkNotEmpty(len: usize) Error!void {
    if (len == 0) return error.InvalidDimension;
}

fn actualPositive(y: f64) bool {
    return y >= 0.5;
}

/// confusion_matrix computes the confusion matrix for binary classification.
/// Returns a 2x2 matrix: [[TN, FP], [FN, TP]].
pub fn confusion_matrix(y_true: []const f64, y_pred: []const f64, allocator: std.mem.Allocator) Error!Matrix(usize) {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var cm = try Matrix(usize).init(allocator, 2, 2);
    errdefer cm.deinit(allocator);

    var tn: usize = 0;
    var fp: usize = 0;
    var fn_: usize = 0;
    var tp: usize = 0;

    for (y_true, y_pred) |yt, yp| {
        const a = actualPositive(yt);
        const p = actualPositive(yp);
        if (a and p) {
            tp += 1;
        } else if (a and !p) {
            fn_ += 1;
        } else if (!a and p) {
            fp += 1;
        } else {
            tn += 1;
        }
    }

    try cm.set(0, 0, tn);
    try cm.set(0, 1, fp);
    try cm.set(1, 0, fn_);
    try cm.set(1, 1, tp);

    return cm;
}

/// accuracy_score computes the accuracy classification score.
pub fn accuracy_score(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var correct: usize = 0;
    for (y_true, y_pred) |yt, yp| {
        if (actualPositive(yt) == actualPositive(yp)) correct += 1;
    }

    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(y_true.len));
}

/// precision_score computes precision: tp / (tp + fp).
pub fn precision_score(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var tp: usize = 0;
    var fp: usize = 0;
    for (y_true, y_pred) |yt, yp| {
        const a = actualPositive(yt);
        const p = actualPositive(yp);
        if (p) {
            if (a) tp += 1 else fp += 1;
        }
    }

    if (tp + fp == 0) return 0.0;
    return @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fp));
}

/// recall_score computes recall: tp / (tp + fn).
pub fn recall_score(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    var tp: usize = 0;
    var fn_: usize = 0;
    for (y_true, y_pred) |yt, yp| {
        const a = actualPositive(yt);
        const p = actualPositive(yp);
        if (a) {
            if (p) tp += 1 else fn_ += 1;
        }
    }

    if (tp + fn_ == 0) return 0.0;
    return @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fn_));
}

/// f1_score computes the F1 score: 2 * (precision * recall) / (precision + recall).
pub fn f1_score(y_true: []const f64, y_pred: []const f64) Error!f64 {
    const prec = try precision_score(y_true, y_pred);
    const rec = try recall_score(y_true, y_pred);

    if (prec + rec == 0.0) return 0.0;
    return 2.0 * prec * rec / (prec + rec);
}

/// roc_curve computes the Receiver Operating Characteristic (ROC) curve.
pub fn roc_curve(y_true: []const f64, y_score: []const f64, allocator: std.mem.Allocator) Error!RocCurveResult {
    try checkLength(y_true.len, y_score.len);
    try checkNotEmpty(y_true.len);

    var n_pos: usize = 0;
    var n_neg: usize = 0;
    for (y_true) |yt| {
        if (actualPositive(yt)) n_pos += 1 else n_neg += 1;
    }
    if (n_pos == 0 or n_neg == 0) return error.InvalidDimension;

    const sorted_scores = try allocator.alloc(f64, y_score.len);
    errdefer allocator.free(sorted_scores);
    @memcpy(sorted_scores, y_score);
    std.mem.sort(f64, sorted_scores, {}, std.sort.desc(f64));

    var unique_thresholds = try allocator.alloc(f64, sorted_scores.len + 1);
    errdefer allocator.free(unique_thresholds);
    var unique_len: usize = 0;
    unique_thresholds[0] = sorted_scores[0] + 1.0;
    unique_len = 1;
    for (sorted_scores) |t| {
        if (t != unique_thresholds[unique_len - 1]) {
            unique_thresholds[unique_len] = t;
            unique_len += 1;
        }
    }

    var fpr = try allocator.alloc(f64, unique_len);
    errdefer allocator.free(fpr);
    var tpr = try allocator.alloc(f64, unique_len);
    errdefer allocator.free(tpr);
    var thresholds = try allocator.alloc(f64, unique_len);
    errdefer allocator.free(thresholds);

    for (0..unique_len) |i| {
        const threshold = unique_thresholds[i];
        var tp: usize = 0;
        var fp: usize = 0;
        for (y_true, y_score) |yt, ys| {
            if (ys >= threshold) {
                if (actualPositive(yt)) tp += 1 else fp += 1;
            }
        }
        fpr[i] = @as(f64, @floatFromInt(fp)) / @as(f64, @floatFromInt(n_neg));
        tpr[i] = @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(n_pos));
        thresholds[i] = threshold;
    }

    allocator.free(sorted_scores);
    allocator.free(unique_thresholds);

    return .{
        .fpr = fpr,
        .tpr = tpr,
        .thresholds = thresholds,
    };
}

fn auc(x: []f64, y: []f64, allocator: std.mem.Allocator) Error!f64 {
    if (x.len != y.len) return error.ShapeMismatch;
    if (x.len < 2) return error.InvalidDimension;

    var indices = try allocator.alloc(usize, x.len);
    defer allocator.free(indices);
    for (0..x.len) |i| indices[i] = i;

    const Context = struct {
        x: []const f64,
        pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
            return ctx.x[a] < ctx.x[b];
        }
    };
    std.sort.block(usize, indices, Context{ .x = x }, Context.lessThan);

    var area: f64 = 0.0;
    for (1..indices.len) |i| {
        const idx1 = indices[i - 1];
        const idx2 = indices[i];
        const dx = x[idx2] - x[idx1];
        const dy = (y[idx1] + y[idx2]) / 2.0;
        area += dx * dy;
    }

    return @abs(area);
}

/// roc_auc_score computes the Area Under the ROC Curve using the trapezoidal rule.
pub fn roc_auc_score(y_true: []const f64, y_score: []const f64, allocator: std.mem.Allocator) Error!f64 {
    const roc = try roc_curve(y_true, y_score, allocator);
    defer {
        allocator.free(roc.fpr);
        allocator.free(roc.tpr);
        allocator.free(roc.thresholds);
    }
    return try auc(roc.fpr, roc.tpr, allocator);
}

/// precision_recall_curve computes precision-recall pairs for different thresholds.
pub fn precision_recall_curve(y_true: []const f64, y_score: []const f64, allocator: std.mem.Allocator) Error!PrecisionRecallResult {
    try checkLength(y_true.len, y_score.len);
    try checkNotEmpty(y_true.len);

    var n_pos: usize = 0;
    for (y_true) |yt| {
        if (actualPositive(yt)) n_pos += 1;
    }

    const sorted_scores = try allocator.alloc(f64, y_score.len);
    errdefer allocator.free(sorted_scores);
    @memcpy(sorted_scores, y_score);
    std.mem.sort(f64, sorted_scores, {}, std.sort.desc(f64));

    var unique_thresholds = try allocator.alloc(f64, sorted_scores.len);
    errdefer allocator.free(unique_thresholds);
    var unique_len: usize = 0;
    for (sorted_scores) |t| {
        if (unique_len == 0 or t != unique_thresholds[unique_len - 1]) {
            unique_thresholds[unique_len] = t;
            unique_len += 1;
        }
    }

    var precision = try allocator.alloc(f64, unique_len);
    errdefer allocator.free(precision);
    var recall = try allocator.alloc(f64, unique_len);
    errdefer allocator.free(recall);
    var thresholds = try allocator.alloc(f64, unique_len);
    errdefer allocator.free(thresholds);

    for (0..unique_len) |i| {
        const threshold = unique_thresholds[i];
        var tp: usize = 0;
        var fp: usize = 0;
        for (y_true, y_score) |yt, ys| {
            if (ys >= threshold) {
                if (actualPositive(yt)) tp += 1 else fp += 1;
            }
        }
        precision[i] = if (tp + fp > 0) @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fp)) else 1.0;
        recall[i] = if (n_pos > 0) @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(n_pos)) else 0.0;
        thresholds[i] = threshold;
    }

    allocator.free(sorted_scores);
    allocator.free(unique_thresholds);

    return .{
        .precision = precision,
        .recall = recall,
        .thresholds = thresholds,
    };
}

/// average_precision_score computes average precision from prediction scores.
pub fn average_precision_score(y_true: []const f64, y_score: []const f64, allocator: std.mem.Allocator) Error!f64 {
    const pr = try precision_recall_curve(y_true, y_score, allocator);
    defer {
        allocator.free(pr.precision);
        allocator.free(pr.recall);
        allocator.free(pr.thresholds);
    }

    if (pr.recall.len < 2) return 0.0;

    var ap: f64 = 0.0;
    for (1..pr.recall.len) |i| {
        const delta_recall = @abs(pr.recall[i - 1] - pr.recall[i]);
        ap += delta_recall * pr.precision[i];
    }

    return ap;
}

/// log_loss computes the logistic loss (cross-entropy loss).
pub fn log_loss(y_true: []const f64, y_pred: []const f64) Error!f64 {
    try checkLength(y_true.len, y_pred.len);
    try checkNotEmpty(y_true.len);

    const eps = 1e-15;
    var total: f64 = 0.0;
    for (y_true, y_pred) |yt, yp| {
        const p = @max(eps, @min(1.0 - eps, yp));
        total += -(yt * @log(p) + (1.0 - yt) * @log(1.0 - p));
    }

    return total / @as(f64, @floatFromInt(y_true.len));
}

/// gini_coefficient computes the Gini coefficient from prediction scores.
pub fn gini_coefficient(y_true: []const f64, y_score: []const f64, allocator: std.mem.Allocator) Error!f64 {
    const auc_val = try roc_auc_score(y_true, y_score, allocator);
    return 2.0 * auc_val - 1.0;
}

/// ks_statistic computes the Kolmogorov-Smirnov statistic.
pub fn ks_statistic(y_true: []const f64, y_score: []const f64) Error!f64 {
    try checkLength(y_true.len, y_score.len);
    try checkNotEmpty(y_true.len);

    var n_pos: usize = 0;
    var n_neg: usize = 0;
    for (y_true) |yt| {
        if (actualPositive(yt)) n_pos += 1 else n_neg += 1;
    }
    if (n_pos == 0 or n_neg == 0) return error.InvalidDimension;

    var max_ks: f64 = 0.0;
    for (y_score) |threshold| {
        var pos_count: usize = 0;
        var neg_count: usize = 0;
        for (y_true, y_score) |yt, ys| {
            if (ys <= threshold) {
                if (actualPositive(yt)) pos_count += 1 else neg_count += 1;
            }
        }
        const pos_cdf = @as(f64, @floatFromInt(pos_count)) / @as(f64, @floatFromInt(n_pos));
        const neg_cdf = @as(f64, @floatFromInt(neg_count)) / @as(f64, @floatFromInt(n_neg));
        const ks = @abs(pos_cdf - neg_cdf);
        if (ks > max_ks) max_ks = ks;
    }

    return max_ks;
}

test "confusion_matrix" {
    const y_true = &[_]f64{ 0.0, 1.0, 1.0, 0.0, 1.0, 0.0 };
    const y_pred = &[_]f64{ 0.0, 1.0, 0.0, 0.0, 1.0, 1.0 };
    var cm = try confusion_matrix(y_true, y_pred, std.testing.allocator);
    defer cm.deinit(std.testing.allocator);

    try std.testing.expectEqual(2, cm.get(0, 0));
    try std.testing.expectEqual(1, cm.get(0, 1));
    try std.testing.expectEqual(1, cm.get(1, 0));
    try std.testing.expectEqual(2, cm.get(1, 1));
}

test "accuracy_score" {
    const y_true = &[_]f64{ 0.0, 1.0, 1.0, 0.0, 1.0 };
    const y_pred = &[_]f64{ 0.0, 1.0, 0.0, 0.0, 1.0 };
    try std.testing.expectApproxEqAbs(try accuracy_score(y_true, y_pred), 0.8, 1e-12);
}

test "precision_score" {
    const y_true = &[_]f64{ 0.0, 1.0, 1.0, 0.0, 1.0 };
    const y_pred = &[_]f64{ 0.0, 1.0, 0.0, 0.0, 1.0 };
    try std.testing.expectApproxEqAbs(try precision_score(y_true, y_pred), 1.0, 1e-12);
}

test "recall_score" {
    const y_true = &[_]f64{ 0.0, 1.0, 1.0, 0.0, 1.0 };
    const y_pred = &[_]f64{ 0.0, 1.0, 0.0, 0.0, 1.0 };
    try std.testing.expectApproxEqAbs(try recall_score(y_true, y_pred), 2.0 / 3.0, 1e-12);
}

test "f1_score" {
    const y_true = &[_]f64{ 0.0, 1.0, 1.0, 0.0, 1.0 };
    const y_pred = &[_]f64{ 0.0, 1.0, 0.0, 0.0, 1.0 };
    try std.testing.expectApproxEqAbs(try f1_score(y_true, y_pred), 0.8, 1e-12);
}

test "roc_auc_score" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.4, 0.35, 0.8 };
    const auc_val = try roc_auc_score(y_true, y_score, std.testing.allocator);
    try std.testing.expectApproxEqAbs(auc_val, 0.75, 1e-12);
}

test "log_loss" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_pred = &[_]f64{ 0.1, 0.2, 0.7, 0.99 };
    const loss = try log_loss(y_true, y_pred);
    const expected = -(@log(0.9) + @log(0.8) + @log(0.7) + @log(0.99)) / 4.0;
    try std.testing.expectApproxEqAbs(loss, expected, 1e-12);
}

test "gini_coefficient" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.4, 0.35, 0.8 };
    const gini = try gini_coefficient(y_true, y_score, std.testing.allocator);
    try std.testing.expectApproxEqAbs(gini, 0.5, 1e-12);
}

test "ks_statistic" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.4, 0.35, 0.8 };
    const ks = try ks_statistic(y_true, y_score);
    try std.testing.expectApproxEqAbs(ks, 0.5, 1e-12);
}

test "roc_curve" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.4, 0.35, 0.8 };
    const roc = try roc_curve(y_true, y_score, std.testing.allocator);
    defer {
        std.testing.allocator.free(roc.fpr);
        std.testing.allocator.free(roc.tpr);
        std.testing.allocator.free(roc.thresholds);
    }
    try std.testing.expectApproxEqAbs(roc.fpr[0], 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(roc.tpr[0], 0.0, 1e-12);
    try std.testing.expectApproxEqAbs(roc.fpr[roc.fpr.len - 1], 1.0, 1e-12);
    try std.testing.expectApproxEqAbs(roc.tpr[roc.tpr.len - 1], 1.0, 1e-12);
}

test "precision_recall_curve" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.4, 0.35, 0.8 };
    const pr = try precision_recall_curve(y_true, y_score, std.testing.allocator);
    defer {
        std.testing.allocator.free(pr.precision);
        std.testing.allocator.free(pr.recall);
        std.testing.allocator.free(pr.thresholds);
    }
    try std.testing.expectEqual(pr.precision.len, pr.recall.len);
    try std.testing.expectEqual(pr.precision.len, pr.thresholds.len);
}

test "average_precision_score" {
    const y_true = &[_]f64{ 0.0, 0.0, 1.0, 1.0 };
    const y_score = &[_]f64{ 0.1, 0.4, 0.35, 0.8 };
    const ap = try average_precision_score(y_true, y_score, std.testing.allocator);
    try std.testing.expect(ap >= 0.0);
    try std.testing.expect(ap <= 1.0);
}
