const std = @import("std");

const marker_default_size = 6;
const marker_default_opacity = 1.0;
const line_default_width = 2;

pub const TraceType = enum {
    scatter,
    line,
    bar,
    heatmap,
    pie,
    surface,
    box,
    violin,
    histogram,
    histogram2d,
    contour,
    ohlc,
    candlestick,
    waterfall,
    sunburst,
    treemap,
    sankey,
    table,
};

pub const Mode = enum {
    markers,
    lines,
    lines_markers,

    pub fn write_json(self: Mode, jw: *std.json.Stringify) !void {
        const s = switch (self) {
            .markers => "markers",
            .lines => "lines",
            .lines_markers => "lines+markers",
        };
        try jw.write(s);
    }
};

pub const Marker = struct {
    color: []const u8 = "",
    size: f64 = marker_default_size,
    opacity: f64 = marker_default_opacity,

    pub fn is_empty(self: Marker) bool {
        return self.color.len == 0 and self.size == marker_default_size and self.opacity == marker_default_opacity;
    }

    pub fn write_json(self: Marker, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        if (self.color.len > 0) {
            try jw.objectField("color");
            try jw.write(self.color);
        }
        if (self.size != marker_default_size) {
            try jw.objectField("size");
            try jw.write(self.size);
        }
        if (self.opacity != marker_default_opacity) {
            try jw.objectField("opacity");
            try jw.write(self.opacity);
        }
        try jw.endObject();
    }
};

pub const Line = struct {
    color: []const u8 = "",
    width: f64 = line_default_width,
    dash: []const u8 = "",

    pub fn is_empty(self: Line) bool {
        return self.color.len == 0 and self.width == line_default_width and self.dash.len == 0;
    }

    pub fn write_json(self: Line, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        if (self.color.len > 0) {
            try jw.objectField("color");
            try jw.write(self.color);
        }
        if (self.width != line_default_width) {
            try jw.objectField("width");
            try jw.write(self.width);
        }
        if (self.dash.len > 0) {
            try jw.objectField("dash");
            try jw.write(self.dash);
        }
        try jw.endObject();
    }
};

pub const ScatterTrace = struct {
    x: []const f64 = &.{},
    y: []const f64 = &.{},
    name: []const u8 = "",
    mode: Mode = .markers,
    marker: Marker = .{},
    line: Line = .{},

    pub fn write_json(self: ScatterTrace, jw: *std.json.Stringify) !void {
        if (self.x.len != self.y.len) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("scatter");
        try jw.objectField("x");
        try jw.write(self.x);
        try jw.objectField("y");
        try jw.write(self.y);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.objectField("mode");
        try self.mode.write_json(jw);
        if (!self.marker.is_empty()) {
            try jw.objectField("marker");
            try self.marker.write_json(jw);
        }
        if (!self.line.is_empty()) {
            try jw.objectField("line");
            try self.line.write_json(jw);
        }
        try jw.endObject();
    }
};

pub const LineTrace = struct {
    x: []const f64 = &.{},
    y: []const f64 = &.{},
    name: []const u8 = "",
    mode: Mode = .lines,
    marker: Marker = .{},
    line: Line = .{},

    pub fn write_json(self: LineTrace, jw: *std.json.Stringify) !void {
        if (self.x.len != self.y.len) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("scatter");
        try jw.objectField("x");
        try jw.write(self.x);
        try jw.objectField("y");
        try jw.write(self.y);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.objectField("mode");
        try self.mode.write_json(jw);
        if (!self.marker.is_empty()) {
            try jw.objectField("marker");
            try self.marker.write_json(jw);
        }
        if (!self.line.is_empty()) {
            try jw.objectField("line");
            try self.line.write_json(jw);
        }
        try jw.endObject();
    }
};

pub const BarTrace = struct {
    x: []const f64 = &.{},
    y: []const f64 = &.{},
    x_labels: ?[]const []const u8 = null,
    y_labels: ?[]const []const u8 = null,
    name: []const u8 = "",
    orientation: []const u8 = "",
    marker: Marker = .{},

    pub fn write_json(self: BarTrace, jw: *std.json.Stringify) !void {
        if (self.x_labels) |labels| {
            if (labels.len != self.x.len) return error.InvalidDimension;
        }
        if (self.y_labels) |labels| {
            if (labels.len != self.y.len) return error.InvalidDimension;
        }
        const x_dim = if (self.x_labels) |labels| labels.len else self.x.len;
        const y_dim = if (self.y_labels) |labels| labels.len else self.y.len;
        if (x_dim != y_dim) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("bar");
        try jw.objectField("x");
        if (self.x_labels) |labels| {
            try jw.write(labels);
        } else {
            try jw.write(self.x);
        }
        try jw.objectField("y");
        if (self.y_labels) |labels| {
            try jw.write(labels);
        } else {
            try jw.write(self.y);
        }
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        if (self.orientation.len > 0) {
            try jw.objectField("orientation");
            try jw.write(self.orientation);
        }
        if (!self.marker.is_empty()) {
            try jw.objectField("marker");
            try self.marker.write_json(jw);
        }
        try jw.endObject();
    }
};

pub const HeatmapTrace = struct {
    z: []const []const f64 = &.{},
    x: ?[]const []const u8 = null,
    y: ?[]const []const u8 = null,
    colorscale: []const u8 = "Viridis",
    name: []const u8 = "",

    pub fn write_json(self: HeatmapTrace, jw: *std.json.Stringify) !void {
        if (self.z.len > 0) {
            const cols = self.z[0].len;
            for (self.z[1..]) |row| {
                if (row.len != cols) return error.InvalidDimension;
            }
            if (self.x) |labels| {
                if (labels.len != cols) return error.InvalidDimension;
            }
        }
        if (self.y) |labels| {
            if (labels.len != self.z.len) return error.InvalidDimension;
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("heatmap");
        try jw.objectField("z");
        try jw.write(self.z);
        if (self.x) |labels| {
            try jw.objectField("x");
            try jw.write(labels);
        }
        if (self.y) |labels| {
            try jw.objectField("y");
            try jw.write(labels);
        }
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.objectField("colorscale");
        try jw.write(self.colorscale);
        try jw.endObject();
    }
};

pub const PieTrace = struct {
    labels: []const []const u8 = &.{},
    values: []const f64 = &.{},
    name: []const u8 = "",
    hole: f64 = 0,

    pub fn write_json(self: PieTrace, jw: *std.json.Stringify) !void {
        if (self.labels.len != self.values.len) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("pie");
        try jw.objectField("labels");
        try jw.write(self.labels);
        try jw.objectField("values");
        try jw.write(self.values);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        if (self.hole > 0) {
            try jw.objectField("hole");
            try jw.write(self.hole);
        }
        try jw.endObject();
    }
};

pub const SurfaceTrace = struct {
    z: []const []const f64 = &.{},
    x: ?[]const f64 = null,
    y: ?[]const f64 = null,
    colorscale: []const u8 = "Viridis",
    name: []const u8 = "",

    pub fn write_json(self: SurfaceTrace, jw: *std.json.Stringify) !void {
        if (self.z.len > 0) {
            const cols = self.z[0].len;
            for (self.z[1..]) |row| {
                if (row.len != cols) return error.InvalidDimension;
            }
            if (self.x) |xs| {
                if (xs.len != cols) return error.InvalidDimension;
            }
            if (self.y) |ys| {
                if (ys.len != self.z.len) return error.InvalidDimension;
            }
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("surface");
        try jw.objectField("z");
        try jw.write(self.z);
        if (self.x) |xs| {
            try jw.objectField("x");
            try jw.write(xs);
        }
        if (self.y) |ys| {
            try jw.objectField("y");
            try jw.write(ys);
        }
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.objectField("colorscale");
        try jw.write(self.colorscale);
        try jw.endObject();
    }
};

pub const BoxTrace = struct {
    y: []const f64 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: BoxTrace, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("box");
        try jw.objectField("y");
        try jw.write(self.y);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const ViolinTrace = struct {
    y: []const f64 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: ViolinTrace, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("violin");
        try jw.objectField("y");
        try jw.write(self.y);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const HistogramTrace = struct {
    x: []const f64 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: HistogramTrace, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("histogram");
        try jw.objectField("x");
        try jw.write(self.x);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const Histogram2dTrace = struct {
    x: []const f64 = &.{},
    y: []const f64 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: Histogram2dTrace, jw: *std.json.Stringify) !void {
        if (self.x.len != self.y.len) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("histogram2d");
        try jw.objectField("x");
        try jw.write(self.x);
        try jw.objectField("y");
        try jw.write(self.y);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const ContourTrace = struct {
    z: []const []const f64 = &.{},
    x: []const f64 = &.{},
    y: []const f64 = &.{},
    colorscale: []const u8 = "Viridis",
    name: []const u8 = "",

    pub fn write_json(self: ContourTrace, jw: *std.json.Stringify) !void {
        if (self.z.len > 0) {
            const cols = self.z[0].len;
            for (self.z[1..]) |row| {
                if (row.len != cols) return error.InvalidDimension;
            }
        }
        if (self.x.len != self.y.len) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("contour");
        try jw.objectField("z");
        try jw.write(self.z);
        if (self.x.len > 0) {
            try jw.objectField("x");
            try jw.write(self.x);
        }
        if (self.y.len > 0) {
            try jw.objectField("y");
            try jw.write(self.y);
        }
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.objectField("colorscale");
        try jw.write(self.colorscale);
        try jw.endObject();
    }
};

pub const OHLCTrace = struct {
    open: []const f64 = &.{},
    high: []const f64 = &.{},
    low: []const f64 = &.{},
    close: []const f64 = &.{},
    x: []const []const u8 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: OHLCTrace, jw: *std.json.Stringify) !void {
        const n = self.open.len;
        if (self.high.len != n or self.low.len != n or self.close.len != n or self.x.len != n) {
            return error.InvalidDimension;
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("ohlc");
        try jw.objectField("open");
        try jw.write(self.open);
        try jw.objectField("high");
        try jw.write(self.high);
        try jw.objectField("low");
        try jw.write(self.low);
        try jw.objectField("close");
        try jw.write(self.close);
        try jw.objectField("x");
        try jw.write(self.x);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const CandlestickTrace = struct {
    open: []const f64 = &.{},
    high: []const f64 = &.{},
    low: []const f64 = &.{},
    close: []const f64 = &.{},
    x: []const []const u8 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: CandlestickTrace, jw: *std.json.Stringify) !void {
        const n = self.open.len;
        if (self.high.len != n or self.low.len != n or self.close.len != n or self.x.len != n) {
            return error.InvalidDimension;
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("candlestick");
        try jw.objectField("open");
        try jw.write(self.open);
        try jw.objectField("high");
        try jw.write(self.high);
        try jw.objectField("low");
        try jw.write(self.low);
        try jw.objectField("close");
        try jw.write(self.close);
        try jw.objectField("x");
        try jw.write(self.x);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const WaterfallTrace = struct {
    x: []const []const u8 = &.{},
    y: []const f64 = &.{},
    measure: []const []const u8 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: WaterfallTrace, jw: *std.json.Stringify) !void {
        if (self.x.len != self.y.len or self.measure.len != self.y.len) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("waterfall");
        try jw.objectField("x");
        try jw.write(self.x);
        try jw.objectField("y");
        try jw.write(self.y);
        try jw.objectField("measure");
        try jw.write(self.measure);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const SunburstTrace = struct {
    ids: []const []const u8 = &.{},
    labels: []const []const u8 = &.{},
    parents: []const []const u8 = &.{},
    values: []const f64 = &.{},

    pub fn write_json(self: SunburstTrace, jw: *std.json.Stringify) !void {
        const n = self.ids.len;
        if (self.labels.len != n or self.parents.len != n or self.values.len != n) {
            return error.InvalidDimension;
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("sunburst");
        try jw.objectField("ids");
        try jw.write(self.ids);
        try jw.objectField("labels");
        try jw.write(self.labels);
        try jw.objectField("parents");
        try jw.write(self.parents);
        try jw.objectField("values");
        try jw.write(self.values);
        try jw.endObject();
    }
};

pub const TreemapTrace = struct {
    ids: []const []const u8 = &.{},
    labels: []const []const u8 = &.{},
    parents: []const []const u8 = &.{},
    values: []const f64 = &.{},

    pub fn write_json(self: TreemapTrace, jw: *std.json.Stringify) !void {
        const n = self.ids.len;
        if (self.labels.len != n or self.parents.len != n or self.values.len != n) {
            return error.InvalidDimension;
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("treemap");
        try jw.objectField("ids");
        try jw.write(self.ids);
        try jw.objectField("labels");
        try jw.write(self.labels);
        try jw.objectField("parents");
        try jw.write(self.parents);
        try jw.objectField("values");
        try jw.write(self.values);
        try jw.endObject();
    }
};

pub const SankeyNode = struct {
    label: []const []const u8 = &.{},

    pub fn write_json(self: SankeyNode, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("label");
        try jw.write(self.label);
        try jw.endObject();
    }
};

pub const SankeyLink = struct {
    source: []const usize = &.{},
    target: []const usize = &.{},
    value: []const f64 = &.{},

    pub fn write_json(self: SankeyLink, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("source");
        try jw.write(self.source);
        try jw.objectField("target");
        try jw.write(self.target);
        try jw.objectField("value");
        try jw.write(self.value);
        try jw.endObject();
    }
};

pub const SankeyTrace = struct {
    node: SankeyNode = .{},
    link: SankeyLink = .{},
    name: []const u8 = "",

    pub fn write_json(self: SankeyTrace, jw: *std.json.Stringify) !void {
        const n = self.link.source.len;
        if (self.link.target.len != n or self.link.value.len != n) return error.InvalidDimension;

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("sankey");
        try jw.objectField("node");
        try self.node.write_json(jw);
        try jw.objectField("link");
        try self.link.write_json(jw);
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const TableTrace = struct {
    header: []const []const u8 = &.{},
    cells: []const []const []const u8 = &.{},
    name: []const u8 = "",

    pub fn write_json(self: TableTrace, jw: *std.json.Stringify) !void {
        for (self.cells) |row| {
            if (row.len != self.header.len) return error.InvalidDimension;
        }

        try jw.beginObject();
        try jw.objectField("type");
        try jw.write("table");
        try jw.objectField("header");
        try jw.beginObject();
        try jw.objectField("values");
        try jw.write(self.header);
        try jw.endObject();
        try jw.objectField("cells");
        try jw.beginObject();
        try jw.objectField("values");
        try jw.write(self.cells);
        try jw.endObject();
        if (self.name.len > 0) {
            try jw.objectField("name");
            try jw.write(self.name);
        }
        try jw.endObject();
    }
};

pub const Trace = union(TraceType) {
    scatter: ScatterTrace,
    line: LineTrace,
    bar: BarTrace,
    heatmap: HeatmapTrace,
    pie: PieTrace,
    surface: SurfaceTrace,
    box: BoxTrace,
    violin: ViolinTrace,
    histogram: HistogramTrace,
    histogram2d: Histogram2dTrace,
    contour: ContourTrace,
    ohlc: OHLCTrace,
    candlestick: CandlestickTrace,
    waterfall: WaterfallTrace,
    sunburst: SunburstTrace,
    treemap: TreemapTrace,
    sankey: SankeyTrace,
    table: TableTrace,

    pub fn write_json(self: Trace, jw: *std.json.Stringify) !void {
        switch (self) {
            inline else => |trace| try trace.write_json(jw),
        }
    }
};

test "Scatter trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = ScatterTrace{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
        .mode = .lines_markers,
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"scatter\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"mode\":\"lines+markers\"") != null);
}

test "Mismatched x/y returns InvalidDimension" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = ScatterTrace{
        .x = &.{ 1, 2 },
        .y = &.{1},
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Line trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = LineTrace{
        .x = &.{ 0, 1, 2 },
        .y = &.{ 1, 2, 3 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"scatter\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"mode\":\"lines\"") != null);
}

test "Bar trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = BarTrace{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"bar\"") != null);
}

test "Heatmap trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = HeatmapTrace{
        .z = &.{
            &.{ 1, 2 },
            &.{ 3, 4 },
        },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"heatmap\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"colorscale\":\"Viridis\"") != null);
}

test "LineTrace mismatched x/y returns InvalidDimension" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = LineTrace{
        .x = &.{ 1, 2 },
        .y = &.{1},
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "BarTrace mismatched x_labels/x returns InvalidDimension" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = BarTrace{
        .x = &.{ 1, 2, 3 },
        .x_labels = &.{ "a", "b" },
        .y = &.{ 4, 5, 6 },
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "HeatmapTrace mismatched x label length returns InvalidDimension" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = HeatmapTrace{
        .z = &.{
            &.{ 1, 2, 3 },
            &.{ 4, 5, 6 },
        },
        .x = &.{ "a", "b" },
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Trace union dispatches .bar serialization" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = Trace{ .bar = BarTrace{
        .x = &.{ 1, 2 },
        .y = &.{ 3, 4 },
    } };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"bar\"") != null);
}

test "Non-default Marker and Line serialization" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = ScatterTrace{
        .x = &.{ 1, 2 },
        .y = &.{ 3, 4 },
        .marker = .{
            .color = "red",
            .size = 10,
            .opacity = 0.5,
        },
        .line = .{
            .color = "blue",
            .width = 3,
            .dash = "dash",
        },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"color\":\"red\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"size\":10") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"opacity\":0.5") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"color\":\"blue\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"width\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"dash\":\"dash\"") != null);
}

test "Pie trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = PieTrace{
        .labels = &.{ "A", "B", "C" },
        .values = &.{ 10, 20, 30 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"pie\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"labels\":[\"A\",\"B\",\"C\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"values\":[10,20,30]") != null);
}

test "Pie trace rejects mismatched labels/values" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = PieTrace{
        .labels = &.{ "A", "B" },
        .values = &.{10},
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Surface trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = SurfaceTrace{
        .z = &.{
            &.{ 1, 2 },
            &.{ 3, 4 },
        },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"surface\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"z\":[[1,2],[3,4]]") != null);
}

test "Box trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = BoxTrace{
        .y = &.{ 1, 2, 3, 4, 5 },
        .name = "box1",
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"box\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"box1\"") != null);
}

test "Violin trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = ViolinTrace{
        .y = &.{ 1, 2, 3, 4, 5 },
        .name = "violin1",
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"violin\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"name\":\"violin1\"") != null);
}

test "Histogram trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = HistogramTrace{
        .x = &.{ 1, 2, 2, 3, 3, 3 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"histogram\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"x\":[1,2,2,3,3,3]") != null);
}

test "Histogram2d trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = Histogram2dTrace{
        .x = &.{ 1, 2, 2, 3 },
        .y = &.{ 4, 5, 5, 6 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"histogram2d\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"x\":[1,2,2,3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"y\":[4,5,5,6]") != null);
}

test "Histogram2d rejects mismatched x/y" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = Histogram2dTrace{
        .x = &.{ 1, 2 },
        .y = &.{1},
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Contour trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = ContourTrace{
        .z = &.{
            &.{ 1, 2 },
            &.{ 3, 4 },
        },
        .x = &.{ 0, 1 },
        .y = &.{ 0, 1 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"contour\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"z\":[[1,2],[3,4]]") != null);
}

test "OHLC trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = OHLCTrace{
        .open = &.{ 1, 2 },
        .high = &.{ 3, 4 },
        .low = &.{ 0, 1 },
        .close = &.{ 2, 3 },
        .x = &.{ "2024-01-01", "2024-01-02" },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"ohlc\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"open\":[1,2]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"high\":[3,4]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"low\":[0,1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"close\":[2,3]") != null);
}

test "Candlestick trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = CandlestickTrace{
        .open = &.{ 1, 2 },
        .high = &.{ 3, 4 },
        .low = &.{ 0, 1 },
        .close = &.{ 2, 3 },
        .x = &.{ "2024-01-01", "2024-01-02" },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"candlestick\"") != null);
}

test "OHLC rejects mismatched lengths" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = OHLCTrace{
        .open = &.{ 1, 2 },
        .high = &.{3},
        .low = &.{ 0, 1 },
        .close = &.{ 2, 3 },
        .x = &.{ "2024-01-01", "2024-01-02" },
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Waterfall trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = WaterfallTrace{
        .x = &.{ "A", "B", "C" },
        .y = &.{ 10, 20, -5 },
        .measure = &.{ "relative", "relative", "total" },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"waterfall\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"measure\":[\"relative\",\"relative\",\"total\"]") != null);
}

test "Sunburst trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = SunburstTrace{
        .ids = &.{ "A", "B", "C" },
        .labels = &.{ "A", "B", "C" },
        .parents = &.{ "", "A", "A" },
        .values = &.{ 0, 10, 20 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"sunburst\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"parents\":[\"\",\"A\",\"A\"]") != null);
}

test "Treemap trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = TreemapTrace{
        .ids = &.{ "A", "B", "C" },
        .labels = &.{ "A", "B", "C" },
        .parents = &.{ "", "A", "A" },
        .values = &.{ 0, 10, 20 },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"treemap\"") != null);
}

test "Sankey trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = SankeyTrace{
        .node = .{ .label = &.{ "A", "B", "C" } },
        .link = .{
            .source = &.{ 0, 1 },
            .target = &.{ 1, 2 },
            .value = &.{ 8, 4 },
        },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"sankey\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"source\":[0,1]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"target\":[1,2]") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"value\":[8,4]") != null);
}

test "Sankey rejects mismatched link lengths" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = SankeyTrace{
        .link = .{
            .source = &.{ 0, 1 },
            .target = &.{1},
            .value = &.{ 8, 4 },
        },
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Table trace serializes to JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = TableTrace{
        .header = &.{ "Name", "Value" },
        .cells = &.{
            &.{ "A", "1" },
            &.{ "B", "2" },
        },
    };
    try trace.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"table\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"header\":{\"values\":[\"Name\",\"Value\"]}") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"cells\":{\"values\":[[\"A\",\"1\"],[\"B\",\"2\"]]}") != null);
}

test "Table rejects rows with wrong column count" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const trace = TableTrace{
        .header = &.{ "Name", "Value" },
        .cells = &.{
            &.{"A"},
        },
    };
    try std.testing.expectError(error.InvalidDimension, trace.write_json(&jw));
}

test "Trace union dispatches new trace types" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const traces = &[_]Trace{
        .{ .pie = .{ .labels = &.{"A"}, .values = &.{1} } },
        .{ .box = .{ .y = &.{ 1, 2 } } },
        .{ .histogram = .{ .x = &.{ 1, 2 } } },
        .{ .sankey = .{
            .node = .{ .label = &.{ "A", "B" } },
            .link = .{
                .source = &.{0},
                .target = &.{1},
                .value = &.{1},
            },
        } },
    };

    try jw.beginArray();
    for (traces) |t| try t.write_json(&jw);
    try jw.endArray();

    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"pie\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"box\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"histogram\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"sankey\"") != null);
}
