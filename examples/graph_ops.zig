const std = @import("std");
const zsl = @import("zsl");
const graph = zsl.graph;

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    // ------------------------------------------------------------------
    // Legacy VSL-compatible Graph demo (shortest paths on edge lists).
    // ------------------------------------------------------------------
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

    var g_vsl = try graph.GraphVsl.init(edges, weights, verts, &[_]f64{}, allocator);
    defer g_vsl.deinit(allocator);

    var sp = try g_vsl.shortest_paths(.fw, allocator);
    defer sp.deinit(allocator);

    const d = try sp.dist.get(0, 3);
    std.debug.print("[GraphVsl] shortest distance 0 -> 3 = {d}\n", .{d});

    const p = try sp.path(0, 3, allocator);
    defer allocator.free(p);
    std.debug.print("[GraphVsl] shortest path 0 -> 3 = {any}\n", .{p});

    // ------------------------------------------------------------------
    // New generic Graph(T) demo.
    // ------------------------------------------------------------------
    var g = try graph.Graph(f64).init(allocator, 4);
    defer g.deinit(allocator);

    try g.add_edge(0, 1, 5.0);
    try g.add_edge(1, 2, 3.0);
    try g.add_edge(2, 3, 1.0);
    try g.add_edge(0, 3, 10.0);

    std.debug.print("[Graph(f64)] vertices: {d}, edges: {d}\n", .{ g.nverts, g.n_edges() });

    const nb = try g.neighbors(allocator, 0);
    defer allocator.free(nb);
    std.debug.print("[Graph(f64)] neighbors of 0 = {any}\n", .{nb});

    const dist = try g.floyd_warshall(allocator);
    defer {
        for (dist) |row| allocator.free(row);
        allocator.free(dist);
    }
    std.debug.print("[Graph(f64)] floyd_warshall distance 0 -> 3 = {d}\n", .{try graph.calc_dist(f64, dist, 0, 3)});

    var res = try g.dijkstra(allocator, 0);
    defer res.deinit(allocator);
    std.debug.print("[Graph(f64)] dijkstra distances from 0 = {any}\n", .{res.dist});

    const path_03 = try graph.path(allocator, res.prev, 0, 3);
    defer allocator.free(path_03);
    std.debug.print("[Graph(f64)] reconstructed path 0 -> 3 = {any}\n", .{path_03});
}
