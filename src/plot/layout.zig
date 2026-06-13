const std = @import("std");

pub const Layout = struct {
    title: []const u8 = "",
    width: usize = 700,
    height: usize = 500,
    x_axis: Axis = .{},
    y_axis: Axis = .{},
    annotations: []const Annotation = &.{},

    pub fn write_json(self: Layout, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("title");
        try jw.write(self.title);
        try jw.objectField("width");
        try jw.write(self.width);
        try jw.objectField("height");
        try jw.write(self.height);
        try jw.objectField("xaxis");
        try self.x_axis.write_json(jw);
        try jw.objectField("yaxis");
        try self.y_axis.write_json(jw);
        if (self.annotations.len > 0) {
            try jw.objectField("annotations");
            try jw.beginArray();
            for (self.annotations) |a| try a.write_json(jw);
            try jw.endArray();
        }
        try jw.endObject();
    }
};

pub const Axis = struct {
    title: []const u8 = "",
    range: ?[2]f64 = null,

    pub fn write_json(self: Axis, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("title");
        try jw.write(self.title);
        if (self.range) |r| {
            try jw.objectField("range");
            try jw.write(r);
        }
        try jw.endObject();
    }
};

pub const Annotation = struct {
    text: []const u8 = "",
    x: f64 = 0,
    y: f64 = 0,
    show_arrow: bool = false,

    pub fn write_json(self: Annotation, jw: *std.json.Stringify) !void {
        try jw.beginObject();
        try jw.objectField("text");
        try jw.write(self.text);
        try jw.objectField("x");
        try jw.write(self.x);
        try jw.objectField("y");
        try jw.write(self.y);
        try jw.objectField("showarrow");
        try jw.write(self.show_arrow);
        try jw.endObject();
    }
};

test "Layout JSON omits empty defaults" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const layout = Layout{};
    try layout.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"annotations\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"title\":\"\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"width\":700") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"height\":500") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"xaxis\":{\"title\":\"\"}") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"yaxis\":{\"title\":\"\"}") != null);
}

test "Layout JSON contains title and axis" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const layout = Layout{
        .title = "Demo",
        .width = 800,
        .height = 600,
        .x_axis = .{ .title = "X", .range = .{ 0, 10 } },
        .y_axis = .{ .title = "Y" },
    };
    try layout.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"title\":\"Demo\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"xaxis\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"range\":[0,10]") != null);
}

test "Annotation serialization" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const annotation = Annotation{
        .text = "note",
        .x = 1.5,
        .y = 2.5,
        .show_arrow = true,
    };
    try annotation.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"text\":\"note\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"x\":1.5") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"y\":2.5") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"showarrow\":true") != null);
}

test "Axis with null range omits range field" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

    const axis = Axis{ .title = "X" };
    try axis.write_json(&jw);
    const json = aw.written();
    try std.testing.expect(std.mem.indexOf(u8, json, "\"range\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"title\":\"X\"") != null);
}
