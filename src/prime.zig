const std = @import("std");
const Error = @import("errors.zig").Error;

/// Deterministic primality test for unsigned integers.
pub fn is_prime(p: usize) bool {
    if (p < 2 or p % 2 == 0) {
        return p == 2;
    }
    var i: usize = 3;
    const max = std.math.sqrt(p);
    while (i <= max) : (i += 2) {
        if (p % i == 0) {
            return false;
        }
    }
    return true;
}

/// Sieve of Eratosthenes returning all primes strictly less than `range`.
/// Caller owns the returned slice.
pub fn prime_sieve(range: usize, allocator: std.mem.Allocator) Error![]usize {
    if (range <= 1) {
        return error.InvalidDimension;
    }

    const number_list = try allocator.alloc(bool, range);
    defer allocator.free(number_list);

    for (0..range) |idx| {
        number_list[idx] = (idx % 2 != 0);
    }
    number_list[0] = false;
    number_list[1] = false;
    if (range > 2) {
        number_list[2] = true;
    }

    const limit = std.math.sqrt(range);
    var i: usize = 3;
    while (i <= limit) : (i += 2) {
        if (number_list[i]) {
            var j = i * i;
            while (j < range) : (j += i) {
                number_list[j] = false;
            }
        }
    }

    var count: usize = 0;
    for (number_list) |is_p| {
        if (is_p) count += 1;
    }

    const primes = try allocator.alloc(usize, count);
    var idx: usize = 0;
    for (number_list, 0..) |is_p, n| {
        if (is_p) {
            primes[idx] = n;
            idx += 1;
        }
    }

    return primes;
}

test "is_prime basic cases" {
    try std.testing.expect(!is_prime(0));
    try std.testing.expect(!is_prime(1));
    try std.testing.expect(is_prime(2));
    try std.testing.expect(is_prime(3));
    try std.testing.expect(!is_prime(4));
    try std.testing.expect(is_prime(5));
    try std.testing.expect(is_prime(29));
    try std.testing.expect(!is_prime(30));
}

test "prime_sieve up to 30" {
    const allocator = std.testing.allocator;
    const primes = try prime_sieve(30, allocator);
    defer allocator.free(primes);
    const expected = &[_]usize{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };
    try std.testing.expectEqualSlices(usize, expected, primes);
}

test "prime_sieve rejects invalid range" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(error.InvalidDimension, prime_sieve(0, allocator));
    try std.testing.expectError(error.InvalidDimension, prime_sieve(1, allocator));
}

test "prime_sieve small ranges" {
    const allocator = std.testing.allocator;

    const two = try prime_sieve(2, allocator);
    defer allocator.free(two);
    try std.testing.expectEqualSlices(usize, &[_]usize{}, two);

    const three = try prime_sieve(3, allocator);
    defer allocator.free(three);
    try std.testing.expectEqualSlices(usize, &[_]usize{2}, three);

    const four = try prime_sieve(4, allocator);
    defer allocator.free(four);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 3 }, four);
}
