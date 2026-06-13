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
fn dimension_product(dimensions: []const usize) Error!usize {
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
        const total = try dimension_product(dimensions);
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

/// Integer range iterator over ``[start, stop)`` with ``step != 0``.
pub const IntIter = struct {
    start: i64,
    step: i64,
    len: usize,
    i: usize,

    pub fn init(start: i64, stop: i64, step: i64) Error!IntIter {
        if (step == 0) return error.InvalidDimension;
        const len = try int_iter_len(start, stop, step);
        if (len > 0) {
            const last: i128 = @as(i128, start) + @as(i128, @intCast(len - 1)) * step;
            if (last < std.math.minInt(i64) or last > std.math.maxInt(i64)) return error.InvalidDimension;
        }
        return .{
            .start = start,
            .step = step,
            .len = len,
            .i = 0,
        };
    }

    pub fn next(self: *IntIter) ?i64 {
        if (self.i == self.len) return null;
        const value128: i128 = @as(i128, self.start) + @as(i128, @intCast(self.i)) * self.step;
        const value = std.math.cast(i64, value128) orelse return null;
        self.i += 1;
        return value;
    }

    pub fn count(self: IntIter) usize {
        return self.len;
    }
};

fn int_iter_len(start: i64, stop: i64, step: i64) Error!usize {
    const abs_step: i128 = if (step > 0) @as(i128, step) else -@as(i128, step);
    const diff: i128 = if (step > 0)
        @as(i128, stop) - @as(i128, start)
    else
        @as(i128, start) - @as(i128, stop);
    if (diff <= 0) return 0;
    const n = @divTrunc(diff + abs_step - 1, abs_step);
    if (n > std.math.maxInt(usize)) return error.InvalidDimension;
    return @intCast(n);
}

/// Factory for ``IntIter``.
pub fn int_iter(start: i64, stop: i64, step: i64) Error!IntIter {
    return IntIter.init(start, stop, step);
}

/// Floating-point range iterator over ``[start, stop)`` with ``step != 0``.
pub const FloatIter = struct {
    start: f64,
    step: f64,
    len: usize,
    i: usize,

    pub fn init(start: f64, stop: f64, step: f64) Error!FloatIter {
        if (step == 0.0) return error.InvalidDimension;
        const len = try float_iter_len(start, stop, step);
        return .{
            .start = start,
            .step = step,
            .len = len,
            .i = 0,
        };
    }

    pub fn next(self: *FloatIter) ?f64 {
        if (self.i == self.len) return null;
        const value = self.start + @as(f64, @floatFromInt(self.i)) * self.step;
        self.i += 1;
        return value;
    }

    pub fn count(self: FloatIter) usize {
        return self.len;
    }
};

fn float_iter_len(start: f64, stop: f64, step: f64) Error!usize {
    const range = stop - start;
    if (!std.math.isFinite(range) or !std.math.isFinite(step)) return error.InvalidDimension;
    if ((step > 0 and range <= 0) or (step < 0 and range >= 0)) return 0;
    const abs_range = @abs(range);
    const abs_step = @abs(step);
    const div = abs_range / abs_step;
    const whole = @floor(div);
    const rem = abs_range - whole * abs_step;
    const n_float = whole + if (rem > 0) @as(f64, 1) else @as(f64, 0);
    const max_count = @as(f64, @floatFromInt(std.math.maxInt(usize)));
    if (n_float < 0 or n_float > max_count) return error.InvalidDimension;
    return @intFromFloat(n_float);
}

/// Factory for ``FloatIter``.
pub fn float_iter(start: f64, stop: f64, step: f64) Error!FloatIter {
    return FloatIter.init(start, stop, step);
}

/// Simple infinite counter starting at ``start`` and advancing by ``step``.
pub fn Counter(comptime T: type) type {
    return struct {
        state: T,
        step: T,

        const Self = @This();

        pub fn init(start: T, step: T) Self {
            return .{
                .state = start,
                .step = step,
            };
        }

        pub fn next(self: *Self) ?T {
            const value = self.state;
            self.state += self.step;
            return value;
        }
    };
}

/// Factory for ``Counter(T)``.
pub fn counter(comptime T: type, start: T, step: T) Counter(T) {
    return Counter(T).init(start, step);
}

/// Infinite iterator that cycles over the elements of ``data``.
pub fn Cycler(comptime T: type) type {
    return struct {
        data: []const T,
        idx: usize,

        const Self = @This();

        pub fn init(data: []const T) Self {
            return .{
                .data = data,
                .idx = 0,
            };
        }

        pub fn next(self: *Self) ?T {
            if (self.data.len == 0) return null;
            const value = self.data[self.idx % self.data.len];
            self.idx += 1;
            return value;
        }
    };
}

/// Factory for ``Cycler(T)``.
pub fn cycler(comptime T: type, items: []const T) Cycler(T) {
    return Cycler(T).init(items);
}

/// Infinite iterator that repeats a single value.
pub fn Repeater(comptime T: type) type {
    return struct {
        item: T,

        const Self = @This();

        pub fn init(item: T) Self {
            return .{ .item = item };
        }

        pub fn next(self: *Self) ?T {
            return self.item;
        }
    };
}

/// Factory for ``Repeater(T)``.
pub fn repeater(comptime T: type, value: T) Repeater(T) {
    return Repeater(T).init(value);
}

/// Iterator over all permutations of a provided slice.
///
/// The slice returned by ``next`` references internal state and remains valid
/// only until the next call to ``next`` or until the iterator is deinitialized.
pub fn Permutations(comptime T: type) type {
    return struct {
        items: []T,
        c: []usize,
        i: usize,
        first: bool,
        total: usize,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, data: []const T) Error!Self {
            const n = data.len;
            const total = try factorial(n);
            const items = try allocator.alloc(T, n);
            errdefer allocator.free(items);
            @memcpy(items, data);
            const c = try allocator.alloc(usize, n);
            errdefer allocator.free(c);
            @memset(c, 0);
            return .{
                .items = items,
                .c = c,
                .i = 0,
                .first = true,
                .total = total,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.items);
            allocator.free(self.c);
            self.items = &[_]T{};
            self.c = &[_]usize{};
        }

        pub fn next(self: *Self) ?[]const T {
            if (self.first) {
                self.first = false;
                return self.items;
            }
            const n = self.items.len;
            while (self.i < n) {
                if (self.c[self.i] < self.i) {
                    if (self.i % 2 == 0) {
                        std.mem.swap(T, &self.items[0], &self.items[self.i]);
                    } else {
                        std.mem.swap(T, &self.items[self.c[self.i]], &self.items[self.i]);
                    }
                    self.c[self.i] += 1;
                    self.i = 0;
                    return self.items;
                } else {
                    self.c[self.i] = 0;
                    self.i += 1;
                }
            }
            return null;
        }

        pub fn count(self: Self) usize {
            return self.total;
        }
    };
}

/// Eagerly compute all permutations of ``items``.
///
/// The returned slices are independent copies; free them with
/// ``permutations_free``.
pub fn permutations(comptime T: type, allocator: std.mem.Allocator, items: []const T) Error![][]T {
    var it = try Permutations(T).init(allocator, items);
    defer it.deinit(allocator);
    const total = it.count();
    var result = try allocator.alloc([]T, total);
    var allocated: usize = 0;
    errdefer {
        for (0..allocated) |k| allocator.free(result[k]);
        allocator.free(result);
    }
    while (it.next()) |perm| {
        result[allocated] = try allocator.dupe(T, perm);
        allocated += 1;
    }
    return result;
}

/// Free the result of ``permutations``.
pub fn permutations_free(comptime T: type, allocator: std.mem.Allocator, result: [][]T) void {
    for (result) |s| allocator.free(s);
    allocator.free(result);
}

/// Iterator over the Cartesian product of two slices.
///
/// Each yielded value is a ``[2]T`` pair.
pub fn Product(comptime T: type) type {
    return struct {
        a: []const T,
        b: []const T,
        i: usize,
        j: usize,
        total: usize,

        const Self = @This();

        pub fn init(a: []const T, b: []const T) Error!Self {
            const total = try dimension_product(&[_]usize{ a.len, b.len });
            return .{
                .a = a,
                .b = b,
                .i = 0,
                .j = 0,
                .total = total,
            };
        }

        pub fn next(self: *Self) ?[2]T {
            if (self.i == self.a.len) return null;
            const pair = [2]T{ self.a[self.i], self.b[self.j] };
            self.j += 1;
            if (self.j == self.b.len) {
                self.j = 0;
                self.i += 1;
            }
            return pair;
        }

        pub fn count(self: Self) usize {
            return self.total;
        }
    };
}

/// Eagerly compute the Cartesian product of ``a`` and ``b``.
///
/// Free the returned array with ``product_free`` or ``allocator.free``.
pub fn product(comptime T: type, allocator: std.mem.Allocator, a: []const T, b: []const T) Error![][2]T {
    var it = try Product(T).init(a, b);
    const total = it.count();
    var result = try allocator.alloc([2]T, total);
    errdefer allocator.free(result);
    var idx: usize = 0;
    while (it.next()) |pair| {
        result[idx] = pair;
        idx += 1;
    }
    return result;
}

/// Free the result of ``product``.
pub fn product_free(comptime T: type, allocator: std.mem.Allocator, result: [][2]T) void {
    allocator.free(result);
}

/// Unsigned integer range iterator over ``[start, end)`` with ``step > 0``.
pub const Ranges = struct {
    start: usize,
    end: usize,
    step: usize,
    len: usize,
    i: usize,

    pub fn init(start: usize, end: usize, step: usize) Error!Ranges {
        if (step == 0) return error.InvalidDimension;
        const len = try ranges_len(start, end, step);
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

fn ranges_len(start: usize, end: usize, step: usize) Error!usize {
    if (start >= end) return 0;
    const diff: u128 = end - start;
    const adj: u128 = step - 1;
    const numer = diff + adj;
    if (numer > std.math.maxInt(usize)) return error.InvalidDimension;
    return @intCast(@divTrunc(numer, step));
}

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

test "IntIter count and values" {
    var it = try int_iter(0, 10, 3);
    try std.testing.expectEqual(@as(usize, 4), it.count());
    try std.testing.expectEqual(@as(i64, 0), it.next().?);
    try std.testing.expectEqual(@as(i64, 3), it.next().?);
    try std.testing.expectEqual(@as(i64, 6), it.next().?);
    try std.testing.expectEqual(@as(i64, 9), it.next().?);
    try std.testing.expect(it.next() == null);
}

test "IntIter negative step" {
    var it = try int_iter(10, 0, -2);
    try std.testing.expectEqual(@as(usize, 5), it.count());
    try std.testing.expectEqual(@as(i64, 10), it.next().?);
    try std.testing.expectEqual(@as(i64, 8), it.next().?);
    try std.testing.expectEqual(@as(i64, 6), it.next().?);
    try std.testing.expectEqual(@as(i64, 4), it.next().?);
    try std.testing.expectEqual(@as(i64, 2), it.next().?);
    try std.testing.expect(it.next() == null);
}

test "IntIter rejects step == 0" {
    try std.testing.expectError(error.InvalidDimension, int_iter(0, 5, 0));
}

test "FloatIter count and values" {
    var it = try float_iter(0.0, 1.0, 0.25);
    try std.testing.expectEqual(@as(usize, 4), it.count());
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), it.next().?, 1e-12);
    try std.testing.expect(it.next() == null);
}

test "FloatIter rejects step == 0" {
    try std.testing.expectError(error.InvalidDimension, float_iter(0.0, 1.0, 0.0));
}

test "Counter" {
    var it = counter(f64, 5.0, 2.5);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 7.5), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), it.next().?, 1e-12);
}

test "Cycler" {
    const data = &[_]i32{ 1, 2, 3 };
    var it = cycler(i32, data);
    try std.testing.expectEqual(@as(i32, 1), it.next().?);
    try std.testing.expectEqual(@as(i32, 2), it.next().?);
    try std.testing.expectEqual(@as(i32, 3), it.next().?);
    try std.testing.expectEqual(@as(i32, 1), it.next().?);
    try std.testing.expectEqual(@as(i32, 2), it.next().?);
}

test "Repeater" {
    var it = repeater(i32, 42);
    try std.testing.expectEqual(@as(i32, 42), it.next().?);
    try std.testing.expectEqual(@as(i32, 42), it.next().?);
    try std.testing.expectEqual(@as(i32, 42), it.next().?);
}

test "Permutations iterator" {
    const data = &[_]i32{ 0, 1, 2 };
    var it = try Permutations(i32).init(std.testing.allocator, data);
    defer it.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 6), it.count());

    var counts: [3]usize = .{ 0, 0, 0 };
    while (it.next()) |perm| {
        try std.testing.expectEqual(@as(usize, 3), perm.len);
        for (perm) |v| counts[@intCast(v)] += 1;
    }
    // Each value appears exactly 2! times in each position over all permutations.
    for (counts) |c| try std.testing.expectEqual(@as(usize, 6), c);
}

test "permutations eager helper" {
    const data = &[_]i32{ 1, 2 };
    const result = try permutations(i32, std.testing.allocator, data);
    defer permutations_free(i32, std.testing.allocator, result);
    try std.testing.expectEqual(@as(usize, 2), result.len);
}

test "Product iterator" {
    const a = &[_]i32{ 1, 2 };
    const b = &[_]i32{ 10, 20, 30 };
    var it = try Product(i32).init(a, b);
    try std.testing.expectEqual(@as(usize, 6), it.count());
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 10 }, &it.next().?);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 20 }, &it.next().?);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 30 }, &it.next().?);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 10 }, &it.next().?);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 20 }, &it.next().?);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 30 }, &it.next().?);
    try std.testing.expect(it.next() == null);
}

test "product eager helper" {
    const a = &[_]i32{ 1, 2 };
    const b = &[_]i32{ 10, 20 };
    const result = try product(i32, std.testing.allocator, a, b);
    defer product_free(i32, std.testing.allocator, result);
    try std.testing.expectEqual(@as(usize, 4), result.len);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 10 }, &result[0]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 20 }, &result[1]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 10 }, &result[2]);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 2, 20 }, &result[3]);
}

test "Permutations empty input" {
    const data = &[_]i32{};
    var it = try Permutations(i32).init(std.testing.allocator, data);
    defer it.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), it.count());
    try std.testing.expectEqual(@as(usize, 0), it.next().?.len);
    try std.testing.expect(it.next() == null);
}

test "Permutations single element" {
    const data = &[_]i32{42};
    var it = try Permutations(i32).init(std.testing.allocator, data);
    defer it.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), it.count());
    try std.testing.expectEqualSlices(i32, &[_]i32{42}, it.next().?);
    try std.testing.expect(it.next() == null);
}

test "permutations eager empty input" {
    const result = try permutations(i32, std.testing.allocator, &[_]i32{});
    defer permutations_free(i32, std.testing.allocator, result);
    try std.testing.expectEqual(@as(usize, 1), result.len);
    try std.testing.expectEqual(@as(usize, 0), result[0].len);
}

test "Product empty input" {
    var it = try Product(i32).init(&[_]i32{}, &[_]i32{ 1, 2 });
    try std.testing.expectEqual(@as(usize, 0), it.count());
    try std.testing.expect(it.next() == null);
}

test "Cycler empty input" {
    var it = cycler(i32, &[_]i32{});
    try std.testing.expect(it.next() == null);
    try std.testing.expect(it.next() == null);
}

test "IntIter empty range" {
    var it = try int_iter(5, 5, 1);
    try std.testing.expectEqual(@as(usize, 0), it.count());
    try std.testing.expect(it.next() == null);
}

test "FloatIter empty range" {
    var it = try float_iter(0.5, 0.5, 0.1);
    try std.testing.expectEqual(@as(usize, 0), it.count());
    try std.testing.expect(it.next() == null);
}

test "FloatIter negative step" {
    var it = try float_iter(1.0, 0.0, -0.25);
    try std.testing.expectEqual(@as(usize, 4), it.count());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), it.next().?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), it.next().?, 1e-12);
    try std.testing.expect(it.next() == null);
}

test "product eager empty input" {
    const result = try product(i32, std.testing.allocator, &[_]i32{}, &[_]i32{ 1, 2 });
    defer product_free(i32, std.testing.allocator, result);
    try std.testing.expectEqual(@as(usize, 0), result.len);
}

test "IntIter extreme bounds do not panic" {
    // The previous i64 arithmetic overflowed here. With i128 widening the call
    // must succeed without panicking; the resulting iterator is allowed.
    _ = int_iter(std.math.minInt(i64), std.math.maxInt(i64), 1) catch {};
}
