const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Reduction operation used by `reduce` and `allreduce`.
pub const Operation = enum {
    sum,
    min,
    max,
};

/// Stub wrapper for the Message Passing Interface (MPI).
///
/// In stub mode (the default), `initialize` returns `error.NotImplemented` and
/// all communication routines return `error.NotImplemented`. `finalize` is a
/// no-op so that stub programs can still follow the normal MPI lifecycle.
pub const Mpi = struct {
    /// Initialises MPI. In stub mode this always fails with `error.NotImplemented`.
    pub fn initialize() Error!void {
        return error.NotImplemented;
    }

    /// Shuts down MPI. In stub mode this is a no-op.
    pub fn finalize() void {
        // no-op in stub mode
    }

    /// Returns the rank of the calling processor. In stub mode this fails.
    pub fn rank() Error!i32 {
        return error.NotImplemented;
    }

    /// Returns the number of processors. In stub mode this fails.
    pub fn size() Error!i32 {
        return error.NotImplemented;
    }

    /// Blocks until all processors reach this point. In stub mode this fails.
    pub fn barrier() Error!void {
        return error.NotImplemented;
    }

    /// Sends `values` to processor `dest_rank`. In stub mode this fails.
    pub fn send(comptime T: type, values: []const T, dest_rank: i32) Error!void {
        _ = values;
        _ = dest_rank;
        return error.NotImplemented;
    }

    /// Receives `values` from processor `source_rank`. In stub mode this fails.
    pub fn recv(comptime T: type, values: []T, source_rank: i32) Error!void {
        _ = values;
        _ = source_rank;
        return error.NotImplemented;
    }

    /// Broadcasts `values` from `root` to all processors. In stub mode this fails.
    pub fn bcast(comptime T: type, values: []T, root: i32) Error!void {
        _ = values;
        _ = root;
        return error.NotImplemented;
    }

    /// Reduces `orig` into `dest` on the root processor using `op`.
    /// In stub mode this fails.
    pub fn reduce(comptime T: type, dest: []T, orig: []const T, op: Operation) Error!void {
        _ = dest;
        _ = orig;
        _ = op;
        return error.NotImplemented;
    }

    /// Reduces `orig` into `dest` on all processors using `op`.
    /// In stub mode this fails.
    pub fn allreduce(comptime T: type, dest: []T, orig: []const T, op: Operation) Error!void {
        _ = dest;
        _ = orig;
        _ = op;
        return error.NotImplemented;
    }
};

test "initialize returns NotImplemented" {
    try std.testing.expectError(error.NotImplemented, Mpi.initialize());
}

test "rank and size return NotImplemented" {
    try std.testing.expectError(error.NotImplemented, Mpi.rank());
    try std.testing.expectError(error.NotImplemented, Mpi.size());
}

test "barrier returns NotImplemented" {
    try std.testing.expectError(error.NotImplemented, Mpi.barrier());
}

test "send and recv return NotImplemented" {
    var buf = [_]f64{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.NotImplemented, Mpi.send(f64, &buf, 1));
    try std.testing.expectError(error.NotImplemented, Mpi.recv(f64, &buf, 0));
}

test "bcast returns NotImplemented" {
    var buf = [_]i32{ 1, 2, 3 };
    try std.testing.expectError(error.NotImplemented, Mpi.bcast(i32, &buf, 0));
}

test "reduce and allreduce return NotImplemented" {
    var orig = [_]f64{ 1.0, 2.0, 3.0 };
    var dest = [_]f64{ 0.0, 0.0, 0.0 };
    try std.testing.expectError(error.NotImplemented, Mpi.reduce(f64, &dest, &orig, .sum));
    try std.testing.expectError(error.NotImplemented, Mpi.allreduce(f64, &dest, &orig, .max));
}

test "finalize is a no-op" {
    Mpi.finalize();
}
