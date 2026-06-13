const std = @import("std");
const zsl = @import("zsl");
const Mpi = zsl.mpi.Mpi;

fn showResult(comptime label: []const u8, result: anytype) void {
    if (result) |_| {
        std.debug.print("{s} succeeded (not expected in stub mode)\n", .{label});
    } else |err| {
        std.debug.print("{s} returned {s} (expected in stub mode)\n", .{ label, @errorName(err) });
    }
}

pub fn main() !void {
    std.debug.print("== MPI wrapper stub demo ==\n", .{});

    // In stub mode initialization is not implemented, so we demonstrate
    // graceful error handling instead of a real MPI run.
    showResult("Mpi.initialize()", Mpi.initialize());
    showResult("Mpi.rank()", Mpi.rank());
    showResult("Mpi.size()", Mpi.size());
    showResult("Mpi.barrier()", Mpi.barrier());

    var buf = [_]f64{ 1.0, 2.0, 3.0 };
    showResult("Mpi.send()", Mpi.send(f64, &buf, 1));
    showResult("Mpi.recv()", Mpi.recv(f64, &buf, 0));
    showResult("Mpi.bcast()", Mpi.bcast(f64, &buf, 0));

    var orig = [_]f64{ 1.0, 2.0, 3.0 };
    var dest = [_]f64{ 0.0, 0.0, 0.0 };
    showResult("Mpi.reduce()", Mpi.reduce(f64, &dest, &orig, .sum));
    showResult("Mpi.allreduce()", Mpi.allreduce(f64, &dest, &orig, .sum));

    // finalize is a no-op in stub mode.
    Mpi.finalize();
    std.debug.print("Mpi.finalize() completed (no-op in stub mode)\n", .{});
}
