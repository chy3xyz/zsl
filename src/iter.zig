const std = @import("std");
const Error = @import("errors.zig").Error;

/// Binomial coefficient ``n choose k``. Returns ``InvalidDimension`` if the
/// result does not fit in a ``usize``.
fn choose(n: usize, k: usize) Error!usize {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;
    const kk = if (k > n - k) n - k else k;
    var res: u128 = 1;
    for (1..kk + 1) |i| {
        res = res * (n - kk + i) / i;
        if (res > std.math.maxInt(usize)) return error.InvalidDimension;
    }
    return @intCast(res);
}

/// Factorial of ``n``. Returns ``InvalidDimension`` if the result does not fit
/// in a ``usize``.
fn factorial(n: usize) Error!usize {
    var res: u128 = 1;
    for (1..n + 1) |i| {
        res *= i;
        if (res > std.math.maxInt(usize)) return error.InvalidDimension;
    }
    return @intCast(res);
}

/// Product of an unsigned slice. Returns ``InvalidDimension`` if the result
/// does not fit in a ``usize``.
fn product(dimensions: []const usize) Error!usize {
    var res: u128 = 1;
    for (dimensions) |d| {
        res *= d;
        if (res > std.math.maxInt(usize)) return error.InvalidDimension;
    }
    return @intCast(res);
}

/// Iterator over combinations of ``n`` items taken ``k`` at a time.
///
/// The slice returned by ``next`` references internal state and remains valid
/// only until the next call to ``next`` or until the iterator is deinitialized.
pub const Comb = struct {
    n: usize,
    k: usize,
    total: usize,
    pos: usize,
    idxs: []usize,

    pub fn init(n: usize, k: usize) Error!Comb {
        if (k > n) return error.InvalidDimension;
        const total = try choose(n, k);
        const idxs = try std.heap.page_allocator.alloc(usize, k);
        for (0..k) |i| idxs[i] = i;
        return .{
            .n = n,
            .k = k,
            .total = total,
            .pos = 0,
            .idxs = idxs,
        };
    }

    pub fn deinit(self: *Comb) void {
        std.heap.page_allocator.free(self.idxs);
        self.idxs = &[_]usize{};
    }

    pub fn next(self: *Comb) ?[]const usize {
        if (self.pos == self.total) return null;
        if (self.k == 0) {
            self.pos += 1;
            return self.idxs;
        }
        if (self.pos == 0) {
            self.pos += 1;
            return self.idxs;
        }
        const n = self.n;
        const k = self.k;
        var i = k;
        while (i > 0) {
            i -= 1;
            if (self.idxs[i] != i + n - k) {
                self.idxs[i] += 1;
                for (i + 1..k) |j| {
                    self.idxs[j] = self.idxs[j - 1] + 1;
                }
                self.pos += 1;
                return self.idxs;
            }
        }
        return null;
    }

    pub fn count(self: Comb) usize {
        return self.total;
    }
};

/// Iterator over all permutations of ``n`` items.
///
/// The slice returned by ``next`` references internal state and remains valid
/// only until the next call to ``next`` or until the iterator is deinitialized.
pub const Perm = struct {
    n: usize,
    total: usize,
    pos: usize,
    idxs: []usize,
    cycles: []usize,

    pub fn init(n: usize, allocator: std.mem.Allocator) Error!Perm {
        const total = try factorial(n);
        const idxs = try allocator.alloc(usize, n);
        errdefer allocator.free(idxs);
        const cycles = try allocator.alloc(usize, n);
        errdefer allocator.free(cycles);
        for (0..n) |i| {
            idxs[i] = i;
            cycles[i] = n - i;
        }
        return .{
            .n = n,
            .total = total,
            .pos = 0,
            .idxs = idxs,
            .cycles = cycles,
        };
    }

    pub fn deinit(self: *Perm, allocator: std.mem.Allocator) void {
        allocator.free(self.idxs);
        allocator.free(self.cycles);
        self.idxs = &[_]usize{};
        self.cycles = &[_]usize{};
    }

    pub fn next(self: *Perm) ?[]const usize {
        if (self.pos == self.total) return null;
        if (self.pos == 0) {
            self.pos += 1;
            return self.idxs;
        }
        const n = self.n;
        var i = n;
        while (i > 0) {
            i -= 1;
            self.cycles[i] -= 1;
            if (self.cycles[i] == 0) {
                const val = self.idxs[i];
                for (i..n - 1) |j| {
                    self.idxs[j] = self.idxs[j + 1];
                }
                self.idxs[n - 1] = val;
                self.cycles[i] = n - i;
            } else {
                const j = self.cycles[i];
                std.mem.swap(usize, &self.idxs[i], &self.idxs[n - j]);
                self.pos += 1;
                return self.idxs;
            }
        }
        return null;
    }

    pub fn count(self: Perm) usize {
        return self.total;
    }
};

/// Cartesian product iterator over ``dimensions`` lengths.
///
/// Each yielded tuple is a slice of indices ``[i_0, i_1, ...]`` where
/// ``0 <= i_d < dimensions[d]``. The slice references internal state and
/// remains valid only until the next call to ``next`` or until the iterator is
/// deinitialized.
pub const Prod = struct {
    dims: []usize,
    total: usize,
    pos: usize,
    state: []usize,

    pub fn init(dimensions: []const usize, allocator: std.mem.Allocator) Error!Prod {
        const total = try product(dimensions);
        const dims = try allocator.dupe(usize, dimensions);
        errdefer allocator.free(dims);
        const state = try allocator.alloc(usize, dimensions.len);
        errdefer allocator.free(state);
        @memset(state, 0);
        return .{
            .dims = dims,
            .total = total,
            .pos = 0,
            .state = state,
        };
    }

    pub fn deinit(self: *Prod, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
        allocator.free(self.state);
        self.dims = &[_]usize{};
        self.state = &[_]usize{};
    }

    pub fn next(self: *Prod) ?[]const usize {
        if (self.pos == self.total) return null;
        var tmp = self.pos;
        var i = self.dims.len;
        while (i > 0) {
            i -= 1;
            const d = self.dims[i];
            self.state[i] = tmp % d;
            tmp /= d;
        }
        self.pos += 1;
        return self.state;
    }

    pub fn count(self: Prod) usize {
        return self.total;
    }
};

/// Integer range iterator over ``[start, end)`` with ``step > 0``.
pub const Ranges = struct {
    start: usize,
    end: usize,
    step: usize,
    len: usize,
    i: usize,

    pub fn init(start: usize, end: usize, step: usize) Error!Ranges {
        if (step == 0) return error.InvalidDimension;
        const len = if (start < end) (end - start + step - 1) / step else 0;
        return .{
            .start = start,
            .end = end,
            .step = step,
            .len = len,
            .i = 0,
        };
    }

    pub fn next(self: *Ranges) ?usize {
        if (self.i == self.len) return null;
        const value = self.start + self.i * self.step;
        self.i += 1;
        return value;
    }

    pub fn count(self: Ranges) usize {
        return self.len;
    }
};

/// Simple infinite counter starting at ``start`` and advancing by ``step``.
pub const InfIters = struct {
    state: usize,
    step: usize,

    pub fn init(start: usize, step: usize) InfIters {
        return .{
            .state = start,
            .step = step,
        };
    }

    pub fn next(self: *InfIters) usize {
        const value = self.state;
        self.state += self.step;
        return value;
    }
};

test "Comb count and first values" {
    var c = try Comb.init(4, 2);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 6), c.count());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, c.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 2 }, c.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 3 }, c.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, c.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3 }, c.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, c.next().?);
    try std.testing.expect(c.next() == null);
}

test "Comb rejects k > n" {
    try std.testing.expectError(error.InvalidDimension, Comb.init(2, 3));
}

test "Perm count and first values" {
    var p = try Perm.init(3, std.testing.allocator);
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 6), p.count());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 2, 1 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 0, 2 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2, 0 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 0, 1 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 1, 0 }, p.next().?);
    try std.testing.expect(p.next() == null);
}

test "Prod count and first values" {
    var p = try Prod.init(&[_]usize{ 2, 3 }, std.testing.allocator);
    defer p.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 6), p.count());
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 0 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 2 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 0 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 1 }, p.next().?);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 2 }, p.next().?);
    try std.testing.expect(p.next() == null);
}

test "Ranges count and first values" {
    var r = try Ranges.init(0, 10, 3);
    try std.testing.expectEqual(@as(usize, 4), r.count());
    try std.testing.expectEqual(@as(usize, 0), r.next().?);
    try std.testing.expectEqual(@as(usize, 3), r.next().?);
    try std.testing.expectEqual(@as(usize, 6), r.next().?);
    try std.testing.expectEqual(@as(usize, 9), r.next().?);
    try std.testing.expect(r.next() == null);
}

test "Ranges rejects step == 0" {
    try std.testing.expectError(error.InvalidDimension, Ranges.init(0, 5, 0));
}

test "InfIters first values" {
    var it = InfIters.init(5, 2);
    try std.testing.expectEqual(@as(usize, 5), it.next());
    try std.testing.expectEqual(@as(usize, 7), it.next());
    try std.testing.expectEqual(@as(usize, 9), it.next());
    try std.testing.expectEqual(@as(usize, 11), it.next());
}
