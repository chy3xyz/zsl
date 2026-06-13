const std = @import("std");
pub const la = @import("la.zig");
const Error = @import("errors.zig").Error;

pub const ShortestPaths = enum {
    fw, // Floyd-Warshall method
    dijkstra, // Dijkstra all-pairs (repeated single-source)
    bfs, // Breadth-first search all-pairs (unweighted shortest path in hops)
};

pub const Graph = struct {
    edges: []const [2]usize,
    weights_e: []const f64,
    verts: la.Matrix(f64),
    weights_v: []const f64,
    dist: la.Matrix(f64),
    next: la.Matrix(isize),
    key2edge: std.AutoHashMap(usize, usize),
    shares: std.AutoHashMap(usize, []usize),

    const Self = @This();
    const inf = std.math.inf(f64);

    pub fn init(
        edges: []const [2]usize,
        weights_e: []const f64,
        verts: la.Matrix(f64),
        weights_v: []const f64,
        allocator: std.mem.Allocator,
    ) Error!Self {
        if (weights_e.len > 0 and weights_e.len != edges.len) return error.ShapeMismatch;
        if (weights_v.len > 0 and weights_v.len != edges.len) return error.ShapeMismatch;

        var key2edge = std.AutoHashMap(usize, usize).init(allocator);
        errdefer key2edge.deinit();

        var share_lists = std.AutoHashMap(usize, std.ArrayList(usize)).init(allocator);
        defer {
            var it = share_lists.valueIterator();
            while (it.next()) |list| list.deinit(allocator);
            share_lists.deinit();
        }

        for (edges, 0..) |edge, k| {
            const i = edge[0];
            const j = edge[1];

            const gop_i = try share_lists.getOrPut(i);
            if (!gop_i.found_existing) gop_i.value_ptr.* = std.ArrayList(usize).empty;
            try gop_i.value_ptr.append(allocator, k);

            const gop_j = try share_lists.getOrPut(j);
            if (!gop_j.found_existing) gop_j.value_ptr.* = std.ArrayList(usize).empty;
            try gop_j.value_ptr.append(allocator, k);

            try key2edge.put(hashEdgeKey(i, j), k);
        }

        var shares = std.AutoHashMap(usize, []usize).init(allocator);
        errdefer {
            var it = shares.valueIterator();
            while (it.next()) |slice| allocator.free(slice.*);
            shares.deinit();
        }

        var it = share_lists.iterator();
        while (it.next()) |entry| {
            const owned = try entry.value_ptr.*.toOwnedSlice(allocator);
            try shares.put(entry.key_ptr.*, owned);
        }

        const nv = shares.count();
        if (nv == 0) return error.InvalidDimension;
        if (verts.rows > 0 and verts.rows != nv) return error.ShapeMismatch;
        if (verts.rows > 0 and verts.cols == 0) return error.InvalidDimension;

        var dist = try la.Matrix(f64).init(allocator, nv, nv);
        errdefer dist.deinit(allocator);
        var next = try la.Matrix(isize).init(allocator, nv, nv);
        errdefer next.deinit(allocator);

        try calcDist(&dist, &next, edges, weights_e, verts);

        return .{
            .edges = edges,
            .weights_e = weights_e,
            .verts = verts,
            .weights_v = weights_v,
            .dist = dist,
            .next = next,
            .key2edge = key2edge,
            .shares = shares,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        self.dist.deinit(allocator);
        self.next.deinit(allocator);

        var it = self.shares.valueIterator();
        while (it.next()) |slice| allocator.free(slice.*);
        self.shares.deinit();
        self.key2edge.deinit();

        self.edges = &[_][2]usize{};
        self.weights_e = &[_]f64{};
        self.weights_v = &[_]f64{};
        self.verts = .{
            .data = &[_]f64{},
            .rows = 0,
            .cols = 0,
            .row_stride = 1,
            .col_stride = 1,
        };
    }

    pub fn nverts(self: *const Self) usize {
        return self.dist.rows;
    }

    pub fn shortest_paths(
        self: *const Self,
        method: ShortestPaths,
        allocator: std.mem.Allocator,
    ) Error!Self {
        var result = try init(
            self.edges,
            self.weights_e,
            self.verts,
            self.weights_v,
            allocator,
        );
        errdefer result.deinit(allocator);

        switch (method) {
            .fw => try shortestPathsFw(&result),
            .dijkstra => try shortestPathsDijkstra(&result, allocator),
            .bfs => try shortestPathsBfs(&result, allocator),
        }

        return result;
    }

    pub fn path(self: *const Self, i: usize, j: usize, allocator: std.mem.Allocator) Error![]usize {
        if (try self.next.get(i, j) < 0) {
            return try allocator.alloc(usize, 0);
        }

        var list: std.ArrayList(usize) = .empty;
        errdefer list.deinit(allocator);

        try list.append(allocator, i);
        var u = i;
        while (u != j) {
            const nxt = try self.next.get(u, j);
            if (nxt < 0) {
                list.deinit(allocator);
                return error.NotImplemented;
            }
            u = @intCast(nxt);
            try list.append(allocator, u);
        }

        return try list.toOwnedSlice(allocator);
    }
};

fn hashEdgeKey(i: usize, j: usize) usize {
    return i + 10000001 * j;
}

fn calcDist(
    dist: *la.Matrix(f64),
    next: *la.Matrix(isize),
    edges: []const [2]usize,
    weights_e: []const f64,
    verts: la.Matrix(f64),
) Error!void {
    const nv = dist.rows;

    for (0..nv) |i| {
        for (0..nv) |j| {
            try dist.set(i, j, if (i == j) 0.0 else Graph.inf);
            try next.set(i, j, -1);
        }
    }

    for (edges, 0..) |edge, k| {
        const i = edge[0];
        const j = edge[1];

        var d: f64 = 1.0;
        if (verts.rows > 0) {
            d = 0.0;
            const row_i = try verts.row(i);
            const row_j = try verts.row(j);
            for (0..verts.cols) |dim| {
                const a = try row_i.get(dim);
                const b = try row_j.get(dim);
                const diff = a - b;
                d += diff * diff;
            }
            d = std.math.sqrt(d);
        }
        if (weights_e.len > 0) {
            d *= weights_e[k];
        }
        if (d < 0.0) return error.InvalidDimension;

        try dist.set(i, j, d);
        try next.set(i, j, @intCast(j));
    }
}

fn shortestPathsFw(g: *Graph) Error!void {
    const nv = g.nverts();
    for (0..nv) |k| {
        for (0..nv) |i| {
            for (0..nv) |j| {
                const dik = try g.dist.get(i, k);
                const dkj = try g.dist.get(k, j);
                if (dik == Graph.inf or dkj == Graph.inf) continue;
                const sum = dik + dkj;
                const dij = try g.dist.get(i, j);
                if (sum < dij) {
                    try g.dist.set(i, j, sum);
                    const nik = try g.next.get(i, k);
                    try g.next.set(i, j, nik);
                }
            }
        }
    }
}

fn shortestPathsDijkstra(g: *Graph, allocator: std.mem.Allocator) Error!void {
    const nv = g.nverts();

    var dist_local = try allocator.alloc(f64, nv);
    defer allocator.free(dist_local);
    var visited = try allocator.alloc(bool, nv);
    defer allocator.free(visited);
    var prev = try allocator.alloc(isize, nv);
    defer allocator.free(prev);

    for (0..nv) |s| {
        @memset(dist_local, Graph.inf);
        @memset(visited, false);
        @memset(prev, -1);
        dist_local[s] = 0.0;

        for (0..nv) |_| {
            const u_opt = argminUnvisited(dist_local, visited);
            if (u_opt == null) break;
            const u = u_opt.?;
            if (dist_local[u] == Graph.inf) break;
            visited[u] = true;

            const edge_ids = g.shares.get(u) orelse continue;
            for (edge_ids) |eid| {
                const edge = g.edges[eid];
                if (edge[0] != u) continue;
                const v = edge[1];
                const w = try g.dist.get(u, v);
                if (w == Graph.inf) continue;
                const candidate = dist_local[u] + w;
                if (candidate < dist_local[v]) {
                    dist_local[v] = candidate;
                    prev[v] = @intCast(u);
                }
            }
        }

        for (0..nv) |t| {
            try g.dist.set(s, t, dist_local[t]);
            try g.next.set(s, t, firstHop(s, t, prev));
        }
    }
}

fn shortestPathsBfs(g: *Graph, allocator: std.mem.Allocator) Error!void {
    const nv = g.nverts();

    var prev = try allocator.alloc(isize, nv);
    defer allocator.free(prev);
    var hops = try allocator.alloc(isize, nv);
    defer allocator.free(hops);
    var queue: std.ArrayList(usize) = .empty;
    defer queue.deinit(allocator);

    for (0..nv) |s| {
        @memset(prev, -1);
        @memset(hops, -1);
        queue.clearRetainingCapacity();

        hops[s] = 0;
        try queue.append(allocator, s);
        var qhead: usize = 0;

        while (qhead < queue.items.len) {
            const u = queue.items[qhead];
            qhead += 1;

            const edge_ids = g.shares.get(u) orelse continue;
            for (edge_ids) |eid| {
                const edge = g.edges[eid];
                if (edge[0] != u) continue;
                const v = edge[1];
                if (hops[v] != -1) continue;
                hops[v] = hops[u] + 1;
                prev[v] = @intCast(u);
                try queue.append(allocator, v);
            }
        }

        for (0..nv) |t| {
            const h = hops[t];
            if (h >= 0) {
                try g.dist.set(s, t, @as(f64, @floatFromInt(h)));
            } else {
                try g.dist.set(s, t, Graph.inf);
            }
            try g.next.set(s, t, firstHop(s, t, prev));
        }
    }
}

fn argminUnvisited(dist: []const f64, visited: []const bool) ?usize {
    var best: ?usize = null;
    var best_dist: f64 = Graph.inf;
    for (dist, visited, 0..) |d, vis, i| {
        if (!vis and d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

fn firstHop(s: usize, t: usize, prev: []const isize) isize {
    if (s == t) return -1;
    if (t >= prev.len) return -1;

    var cur = t;
    var parent = prev[cur];
    if (parent < 0) return -1;

    while (parent != @as(isize, @intCast(s))) {
        cur = @intCast(parent);
        if (cur >= prev.len) return -1;
        parent = prev[cur];
        if (parent < 0) return -1;
    }

    return @intCast(cur);
}

test "Floyd-Warshall shortest paths on example graph" {
    const allocator = std.testing.allocator;
    const edges = &[_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 0, 3 } };
    const weights = &[_]f64{ 5.0, 3.0, 1.0, 10.0 };
    const verts = la.Matrix(f64){
        .data = &[_]f64{},
        .rows = 0,
        .cols = 0,
        .row_stride = 1,
        .col_stride = 1,
    };

    var g = try Graph.init(edges, weights, verts, &[_]f64{}, allocator);
    defer g.deinit(allocator);

    var sp = try g.shortest_paths(.fw, allocator);
    defer sp.deinit(allocator);

    try std.testing.expectApproxEqAbs(9.0, try sp.dist.get(0, 3), 1e-12);

    const p = try sp.path(0, 3, allocator);
    defer allocator.free(p);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, p);
}

test "Dijkstra and BFS shortest paths on example graph" {
    const allocator = std.testing.allocator;
    const edges = &[_][2]usize{ .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 0, 3 } };
    const weights = &[_]f64{ 5.0, 3.0, 1.0, 10.0 };
    const verts = la.Matrix(f64){
        .data = &[_]f64{},
        .rows = 0,
        .cols = 0,
        .row_stride = 1,
        .col_stride = 1,
    };

    var g = try Graph.init(edges, weights, verts, &[_]f64{}, allocator);
    defer g.deinit(allocator);

    var sp_d = try g.shortest_paths(.dijkstra, allocator);
    defer sp_d.deinit(allocator);
    try std.testing.expectApproxEqAbs(9.0, try sp_d.dist.get(0, 3), 1e-12);

    const p = try sp_d.path(0, 3, allocator);
    defer allocator.free(p);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, p);

    var sp_b = try g.shortest_paths(.bfs, allocator);
    defer sp_b.deinit(allocator);
    // BFS counts hops, so the direct edge 0 -> 3 is one hop.
    try std.testing.expectApproxEqAbs(1.0, try sp_b.dist.get(0, 3), 1e-12);
}
