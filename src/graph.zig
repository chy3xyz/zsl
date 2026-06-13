const std = @import("std");
pub const la = @import("la.zig");
const util = @import("util.zig");
const Error = @import("errors.zig").Error;

pub const ShortestPaths = enum {
    fw, // Floyd-Warshall method
    dijkstra, // Dijkstra all-pairs (repeated single-source)
    bfs, // Breadth-first search all-pairs (unweighted shortest path in hops)
};

/// VSL-compatible Graph that derives vertex count from the supplied edges.
/// Kept for backward compatibility with the original VSL port.
pub const GraphVsl = struct {
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

    /// Return the edge at index `idx` together with its weight, or null if
    /// the index is out of bounds.
    pub fn get_edge(self: GraphVsl, idx: usize) ?Edge(f64) {
        if (idx >= self.edges.len) return null;
        const e = self.edges[idx];
        const w = if (self.weights_e.len > 0) self.weights_e[idx] else 1.0;
        return Edge(f64){ .u = e[0], .v = e[1], .weight = w };
    }

    /// Return an allocated slice containing the vertices adjacent to `u`
    /// (outgoing neighbours). Caller must free with `allocator.free`.
    pub fn get_adj(self: GraphVsl, allocator: std.mem.Allocator, u: usize) Error![]usize {
        if (u >= self.nverts()) return error.IndexOutOfBounds;

        var list: std.ArrayList(usize) = .empty;
        errdefer list.deinit(allocator);

        const edge_ids = self.shares.get(u) orelse return try list.toOwnedSlice(allocator);
        for (edge_ids) |eid| {
            const e = self.edges[eid];
            if (e[0] == u) {
                try list.append(allocator, e[1]);
            }
        }

        return try list.toOwnedSlice(allocator);
    }

    /// Return an allocated string representation of the distance matrix.
    /// Caller must free with `allocator.free`.
    pub fn str_dist_matrix(self: GraphVsl, allocator: std.mem.Allocator) Error![]const u8 {
        const nv = self.nverts();
        var list: std.ArrayList(u8) = .empty;
        errdefer list.deinit(allocator);

        try list.append(allocator, '[');
        for (0..nv) |i| {
            if (i > 0) try list.appendSlice(allocator, ", ");
            try list.append(allocator, '[');
            for (0..nv) |j| {
                if (j > 0) try list.appendSlice(allocator, ", ");
                const d = try self.dist.get(i, j);
                if (d == Self.inf) {
                    try list.appendSlice(allocator, "inf");
                } else {
                    const s = try std.fmt.allocPrint(allocator, "{d}", .{d});
                    defer allocator.free(s);
                    try list.appendSlice(allocator, s);
                }
            }
            try list.append(allocator, ']');
        }
        try list.append(allocator, ']');

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
            try dist.set(i, j, if (i == j) 0.0 else GraphVsl.inf);
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

fn shortestPathsFw(g: *GraphVsl) Error!void {
    const nv = g.nverts();
    for (0..nv) |k| {
        for (0..nv) |i| {
            for (0..nv) |j| {
                const dik = try g.dist.get(i, k);
                const dkj = try g.dist.get(k, j);
                if (dik == GraphVsl.inf or dkj == GraphVsl.inf) continue;
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

fn shortestPathsDijkstra(g: *GraphVsl, allocator: std.mem.Allocator) Error!void {
    const nv = g.nverts();

    var dist_local = try allocator.alloc(f64, nv);
    defer allocator.free(dist_local);
    var visited = try allocator.alloc(bool, nv);
    defer allocator.free(visited);
    var prev = try allocator.alloc(isize, nv);
    defer allocator.free(prev);

    for (0..nv) |s| {
        @memset(dist_local, GraphVsl.inf);
        @memset(visited, false);
        @memset(prev, -1);
        dist_local[s] = 0.0;

        for (0..nv) |_| {
            const u_opt = argminUnvisited(dist_local, visited);
            if (u_opt == null) break;
            const u = u_opt.?;
            if (dist_local[u] == GraphVsl.inf) break;
            visited[u] = true;

            const edge_ids = g.shares.get(u) orelse continue;
            for (edge_ids) |eid| {
                const edge = g.edges[eid];
                if (edge[0] != u) continue;
                const v = edge[1];
                const w = try g.dist.get(u, v);
                if (w == GraphVsl.inf) continue;
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

fn shortestPathsBfs(g: *GraphVsl, allocator: std.mem.Allocator) Error!void {
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
                try g.dist.set(s, t, GraphVsl.inf);
            }
            try g.next.set(s, t, firstHop(s, t, prev));
        }
    }
}

fn argminUnvisited(dist: []const f64, visited: []const bool) ?usize {
    var best: ?usize = null;
    var best_dist: f64 = GraphVsl.inf;
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

/// Edge in a weighted graph.
pub fn Edge(comptime T: type) type {
    _ = util.Float(T);
    return struct {
        u: usize,
        v: usize,
        weight: T,
    };
}

/// Generic weighted directed graph stored as an adjacency list of edges.
pub fn Graph(comptime T: type) type {
    _ = util.Float(T);

    return struct {
        nverts: usize,
        edges: std.ArrayList(Edge(T)),
        allocator: std.mem.Allocator,

        const Self = @This();
        const inf = std.math.inf(T);

        /// Initialise an empty graph with `nverts` vertices.
        pub fn init(allocator: std.mem.Allocator, nverts: usize) Error!Self {
            return .{
                .nverts = nverts,
                .edges = std.ArrayList(Edge(T)).empty,
                .allocator = allocator,
            };
        }

        /// Release all graph resources.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            _ = allocator;
            self.edges.deinit(self.allocator);
            self.nverts = 0;
        }

        /// Add a directed edge from `u` to `v` with the given weight.
        pub fn add_edge(self: *Self, u: usize, v: usize, weight: T) Error!void {
            if (u >= self.nverts or v >= self.nverts) return error.IndexOutOfBounds;
            try self.edges.append(self.allocator, .{ .u = u, .v = v, .weight = weight });
        }

        /// Number of edges in the graph.
        pub fn n_edges(self: Self) usize {
            return self.edges.items.len;
        }

        /// Return the edge at index `idx`.
        pub fn get_edge(self: Self, idx: usize) Error!Edge(T) {
            if (idx >= self.edges.items.len) return error.IndexOutOfBounds;
            return self.edges.items[idx];
        }

        /// Return an allocated slice with the neighbours of vertex `u`.
        /// Caller must free the returned slice with `allocator.free`.
        pub fn neighbors(self: Self, allocator: std.mem.Allocator, u: usize) Error![]usize {
            if (u >= self.nverts) return error.IndexOutOfBounds;

            var list: std.ArrayList(usize) = .empty;
            errdefer list.deinit(allocator);

            for (self.edges.items) |edge| {
                if (edge.u == u) {
                    try list.append(allocator, edge.v);
                }
            }

            return try list.toOwnedSlice(allocator);
        }

        /// Build an adjacency matrix of shortest-path distances using the
        /// Floyd-Warshall algorithm. The returned slice-of-slices must be freed
        /// by the caller: free each `dist[i]` then free `dist`.
        pub fn floyd_warshall(self: *const Self, allocator: std.mem.Allocator) Error![][]T {
            const n = self.nverts;

            var dist = try allocator.alloc([]T, n);
            errdefer allocator.free(dist);

            var initialized: usize = 0;
            errdefer {
                for (0..initialized) |k| allocator.free(dist[k]);
            }

            for (0..n) |i| {
                dist[i] = try allocator.alloc(T, n);
                initialized += 1;
                for (0..n) |j| {
                    dist[i][j] = if (i == j) 0.0 else Self.inf;
                }
            }

            for (self.edges.items) |edge| {
                dist[edge.u][edge.v] = edge.weight;
            }

            for (0..n) |k| {
                for (0..n) |i| {
                    for (0..n) |j| {
                        const dik = dist[i][k];
                        const dkj = dist[k][j];
                        if (dik == Self.inf or dkj == Self.inf) continue;
                        const sum = dik + dkj;
                        if (sum < dist[i][j]) {
                            dist[i][j] = sum;
                        }
                    }
                }
            }

            return dist;
        }

        /// Result of a single-source Dijkstra search.
        pub const DijkstraResult = struct {
            dist: []T,
            prev: []isize,

            /// Release the result buffers.
            pub fn deinit(self: DijkstraResult, allocator: std.mem.Allocator) void {
                allocator.free(self.dist);
                allocator.free(self.prev);
            }
        };

        /// Run Dijkstra from `src`. Returns distances and parent pointers.
        pub fn dijkstra(self: *const Self, allocator: std.mem.Allocator, src: usize) Error!DijkstraResult {
            if (src >= self.nverts) return error.IndexOutOfBounds;
            const n = self.nverts;

            var dist = try allocator.alloc(T, n);
            errdefer allocator.free(dist);
            var prev = try allocator.alloc(isize, n);
            errdefer allocator.free(prev);
            var visited = try allocator.alloc(bool, n);
            defer allocator.free(visited);

            @memset(dist, Self.inf);
            @memset(prev, -1);
            @memset(visited, false);
            dist[src] = 0.0;

            for (0..n) |_| {
                const u_opt = argminUnvisitedGeneric(T, dist, visited);
                if (u_opt == null) break;
                const u = u_opt.?;
                if (dist[u] == Self.inf) break;
                visited[u] = true;

                for (self.edges.items) |edge| {
                    if (edge.u != u) continue;
                    const v = edge.v;
                    const candidate = dist[u] + edge.weight;
                    if (candidate < dist[v]) {
                        dist[v] = candidate;
                        prev[v] = @intCast(u);
                    }
                }
            }

            return .{ .dist = dist, .prev = prev };
        }
    };
}

fn argminUnvisitedGeneric(comptime T: type, dist: []const T, visited: []const bool) ?usize {
    var best: ?usize = null;
    var best_dist: T = std.math.inf(T);
    for (dist, visited, 0..) |d, vis, i| {
        if (!vis and d < best_dist) {
            best_dist = d;
            best = i;
        }
    }
    return best;
}

/// Reconstruct the vertex sequence from `src` to `target` using the parent
/// array produced by `dijkstra`. Caller must free the returned slice.
pub fn path(allocator: std.mem.Allocator, prev: []const isize, src: usize, target: usize) Error![]usize {
    if (src >= prev.len or target >= prev.len) return error.IndexOutOfBounds;
    if (target == src) {
        const result = try allocator.alloc(usize, 1);
        result[0] = src;
        return result;
    }

    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(allocator);

    var cur: isize = @intCast(target);
    while (cur >= 0) {
        try list.append(allocator, @intCast(cur));
        if (prev[@intCast(cur)] < 0) break;
        cur = prev[@intCast(cur)];
    }

    // If we never reached src, there is no path.
    if (list.items.len == 0 or list.items[list.items.len - 1] != src) {
        list.deinit(allocator);
        return try allocator.alloc(usize, 0);
    }

    std.mem.reverse(usize, list.items);
    return try list.toOwnedSlice(allocator);
}

/// Convenience helper: distance from `u` to `v` in a distance matrix.
pub fn calc_dist(comptime T: type, dist: []const []const T, u: usize, v: usize) error{IndexOutOfBounds}!T {
    if (u >= dist.len) return error.IndexOutOfBounds;
    if (v >= dist[u].len) return error.IndexOutOfBounds;
    return dist[u][v];
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

    var g = try GraphVsl.init(edges, weights, verts, &[_]f64{}, allocator);
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

    var g = try GraphVsl.init(edges, weights, verts, &[_]f64{}, allocator);
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

test "GraphVsl get_edge and get_adj" {
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

    var g = try GraphVsl.init(edges, weights, verts, &[_]f64{}, allocator);
    defer g.deinit(allocator);

    const e0 = g.get_edge(0);
    try std.testing.expect(e0 != null);
    try std.testing.expectEqual(@as(usize, 0), e0.?.u);
    try std.testing.expectEqual(@as(usize, 1), e0.?.v);
    try std.testing.expectApproxEqAbs(5.0, e0.?.weight, 1e-12);

    try std.testing.expectEqual(@as(?Edge(f64), null), g.get_edge(99));

    const adj0 = try g.get_adj(allocator, 0);
    defer allocator.free(adj0);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3 }, adj0);

    const adj2 = try g.get_adj(allocator, 2);
    defer allocator.free(adj2);
    try std.testing.expectEqualSlices(usize, &[_]usize{3}, adj2);

    try std.testing.expectError(error.IndexOutOfBounds, g.get_adj(allocator, 99));
}

test "GraphVsl str_dist_matrix" {
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

    var g = try GraphVsl.init(edges, weights, verts, &[_]f64{}, allocator);
    defer g.deinit(allocator);

    var sp = try g.shortest_paths(.fw, allocator);
    defer sp.deinit(allocator);

    const s = try sp.str_dist_matrix(allocator);
    defer allocator.free(s);

    try std.testing.expect(s.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, s, "inf") != null);
}

test "Generic Graph init, add_edge and utilities" {
    const allocator = std.testing.allocator;
    var g = try Graph(f64).init(allocator, 4);
    defer g.deinit(allocator);

    try g.add_edge(0, 1, 5.0);
    try g.add_edge(1, 2, 3.0);
    try g.add_edge(2, 3, 1.0);
    try g.add_edge(0, 3, 10.0);

    try std.testing.expectEqual(4, g.nverts);
    try std.testing.expectEqual(4, g.n_edges());

    const e = try g.get_edge(2);
    try std.testing.expectEqual(@as(usize, 2), e.u);
    try std.testing.expectEqual(@as(usize, 3), e.v);
    try std.testing.expectApproxEqAbs(1.0, e.weight, 1e-12);

    const nb = try g.neighbors(allocator, 0);
    defer allocator.free(nb);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 1, 3 }, nb);

    try std.testing.expectError(error.IndexOutOfBounds, g.add_edge(4, 0, 1.0));
}

test "Generic Graph floyd_warshall" {
    const allocator = std.testing.allocator;
    var g = try Graph(f64).init(allocator, 4);
    defer g.deinit(allocator);

    try g.add_edge(0, 1, 5.0);
    try g.add_edge(1, 2, 3.0);
    try g.add_edge(2, 3, 1.0);
    try g.add_edge(0, 3, 10.0);

    const dist = try g.floyd_warshall(allocator);
    defer {
        for (dist) |row| allocator.free(row);
        allocator.free(dist);
    }

    try std.testing.expectApproxEqAbs(9.0, try calc_dist(f64, dist, 0, 3), 1e-12);
    try std.testing.expectApproxEqAbs(4.0, try calc_dist(f64, dist, 1, 3), 1e-12);
    try std.testing.expectApproxEqAbs(std.math.inf(f64), try calc_dist(f64, dist, 3, 0), 1e-12);
}

test "Generic Graph dijkstra and path reconstruction" {
    const allocator = std.testing.allocator;
    var g = try Graph(f64).init(allocator, 4);
    defer g.deinit(allocator);

    try g.add_edge(0, 1, 5.0);
    try g.add_edge(1, 2, 3.0);
    try g.add_edge(2, 3, 1.0);
    try g.add_edge(0, 3, 10.0);

    var res = try g.dijkstra(allocator, 0);
    defer res.deinit(allocator);

    try std.testing.expectApproxEqAbs(0.0, res.dist[0], 1e-12);
    try std.testing.expectApproxEqAbs(5.0, res.dist[1], 1e-12);
    try std.testing.expectApproxEqAbs(9.0, res.dist[3], 1e-12);

    const p = try path(allocator, res.prev, 0, 3);
    defer allocator.free(p);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 0, 1, 2, 3 }, p);

    const src_path = try path(allocator, res.prev, 0, 0);
    defer allocator.free(src_path);
    try std.testing.expectEqualSlices(usize, &[_]usize{0}, src_path);
}
