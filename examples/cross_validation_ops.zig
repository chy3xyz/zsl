const std = @import("std");
const zsl = @import("zsl");

const KFold = zsl.model_selection.cross_validation.KFold;
const ShuffleSplit = zsl.model_selection.cross_validation.ShuffleSplit;
const Error = zsl.errors.Error;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    std.debug.print("=== KFold demo ===\n", .{});
    {
        var kf = try KFold.init(allocator, 12, 3, true, 42);
        defer kf.deinit();

        var fold: usize = 0;
        while (try kf.next()) |split| {
            defer {
                allocator.free(split.train_indices);
                allocator.free(split.test_indices);
            }
            std.debug.print("Fold {d}: train=", .{fold});
            for (split.train_indices) |idx| std.debug.print("{d} ", .{idx});
            std.debug.print(" test=", .{});
            for (split.test_indices) |idx| std.debug.print("{d} ", .{idx});
            std.debug.print("\n", .{});
            fold += 1;
        }
    }

    std.debug.print("\n=== ShuffleSplit demo ===\n", .{});
    {
        var ss = try ShuffleSplit.init(allocator, 20, 4, 0.25, 7);
        defer ss.deinit();

        var iter: usize = 0;
        while (try ss.next()) |split| {
            defer {
                allocator.free(split.train_indices);
                allocator.free(split.test_indices);
            }
            std.debug.print("Split {d}: train_len={d} test_len={d}\n", .{
                iter,
                split.train_indices.len,
                split.test_indices.len,
            });
            iter += 1;
        }
    }
}
