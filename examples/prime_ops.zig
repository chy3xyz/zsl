const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    std.debug.print("is_prime(17) = {any}\n", .{zsl.prime.is_prime(17)});
    std.debug.print("is_prime(18) = {any}\n", .{zsl.prime.is_prime(18)});
    std.debug.print("is_prime(29) = {any}\n", .{zsl.prime.is_prime(29)});
    std.debug.print("is_prime(1)  = {any}\n", .{zsl.prime.is_prime(1)});

    const range: usize = 50;
    const primes = try zsl.prime.prime_sieve(range, allocator);
    std.debug.print("primes below {d} = {any}\n", .{ range, primes });
}
