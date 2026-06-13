const std = @import("std");
const zsl = @import("zsl");
const graph = zsl.graph;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const edges = &[_][2]usize{
        .{ 0, 1 },
        .{ 1, 2 },
        .{ 2, 3 },
        .{ 0, 3 },
    };
    const weights = &[_]f64{ 5.0, 3.0, 1.0, 10.0 };
    const verts = zsl.la.Matrix(f64){
        .data = &[_]f64{},
        .rows = 0,
        .cols = 0,
        .row_stride = 1,
        .col_stride = 1,
    };

    var g = try graph.Graph.init(edges, weights, verts, &[_]f64{}, allocator);
    defer g.deinit(allocator);

    var sp = try g.shortest_paths(.fw, allocator);
    defer sp.deinit(allocator);

    const d = try sp.dist.get(0, 3);
    std.debug.print("shortest distance 0 -> 3 = {d}\n", .{d});

    const p = try sp.path(0, 3, allocator);
    defer allocator.free(p);
    std.debug.print("shortest path 0 -> 3 = {any}\n", .{p});
}
