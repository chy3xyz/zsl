const std = @import("std");

const marker_default_size = 6;
const marker_default_opacity = 1.0;
const line_default_width = 2;

pub const TraceType = enum {
    scatter,
    line,
    bar,
    heatmap,
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

pub const Trace = union(TraceType) {
    scatter: ScatterTrace,
    line: LineTrace,
    bar: BarTrace,
    heatmap: HeatmapTrace,

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
