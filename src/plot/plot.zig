const std = @import("std");
const Layout = @import("layout.zig").Layout;
const Trace = @import("trace.zig").Trace;
const ScatterTrace = @import("trace.zig").ScatterTrace;
const LineTrace = @import("trace.zig").LineTrace;
const BarTrace = @import("trace.zig").BarTrace;
const HeatmapTrace = @import("trace.zig").HeatmapTrace;

const html_template =
    \\<!DOCTYPE html>
    \\<html>
    \\  <head>
    \\    <meta charset="utf-8">
    \\    <title>{s}</title>
    \\    <script src="https://cdn.plot.ly/plotly-2.26.2.min.js"></script>
    \\  </head>
    \\  <body>
    \\    <div id="gd" style="width:{d}px;height:{d}px;"></div>
    \\    <script>
    \\      const data = {s};
    \\      const layout = {s};
    \\      Plotly.newPlot("gd", data, layout);
    \\    </script>
    \\  </body>
    \\</html>
;

fn escape_html(allocator: std.mem.Allocator, s: []const u8) error{OutOfMemory}![]u8 {
    var extra: usize = 0;
    for (s) |c| {
        switch (c) {
            '&' => extra += 4,
            '<', '>' => extra += 3,
            '"' => extra += 5,
            else => {},
        }
    }
    if (extra == 0) return try allocator.dupe(u8, s);
    const result = try allocator.alloc(u8, s.len + extra);
    var i: usize = 0;
    for (s) |c| {
        switch (c) {
            '&' => {
                @memcpy(result[i..][0..5], "&amp;");
                i += 5;
            },
            '<' => {
                @memcpy(result[i..][0..4], "&lt;");
                i += 4;
            },
            '>' => {
                @memcpy(result[i..][0..4], "&gt;");
                i += 4;
            },
            '"' => {
                @memcpy(result[i..][0..6], "&quot;");
                i += 6;
            },
            else => {
                result[i] = c;
                i += 1;
            },
        }
    }
    return result;
}

fn is_script_close_whitespace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == 0x0c;
}

fn is_script_close_prefix(s: []const u8, start: usize) bool {
    return s[start] == '<' and s[start + 1] == '/' and
        ascii_caseless_eql(s[start + 2 .. start + 8], "script");
}

// Returns the index one past the end of the script-close tag if one is found,
// or 0 if this is not a script-close tag.
fn scan_script_close_end(s: []const u8, pos: usize) usize {
    var i = pos;
    while (i < s.len and is_script_close_whitespace(s[i])) i += 1;
    if (i == s.len) return i;
    if (s[i] == '>') return i + 1;
    return 0;
}

fn escape_script_close(allocator: std.mem.Allocator, s: []const u8) error{OutOfMemory}![]u8 {
    // Count potential matches to size the output buffer.
    var count: usize = 0;
    var scan: usize = 0;
    while (scan + 8 <= s.len) {
        if (is_script_close_prefix(s, scan)) {
            const after_tag = scan_script_close_end(s, scan + 8);
            if (after_tag != 0) {
                count += 1;
                scan = after_tag;
            } else {
                scan += 1;
            }
        } else {
            scan += 1;
        }
    }

    if (count == 0) return try allocator.dupe(u8, s);

    // Each match adds one extra byte because '<' becomes '<\'.
    const result = try allocator.alloc(u8, s.len + count);
    var i: usize = 0;
    var src: usize = 0;
    while (src + 8 <= s.len) {
        if (is_script_close_prefix(s, src)) {
            const after_tag = scan_script_close_end(s, src + 8);
            if (after_tag != 0) {
                result[i] = '<';
                i += 1;
                result[i] = '\\';
                i += 1;
                const content = s[src + 1 .. after_tag];
                @memcpy(result[i..][0..content.len], content);
                i += content.len;
                src = after_tag;
            } else {
                result[i] = s[src];
                i += 1;
                src += 1;
            }
        } else {
            result[i] = s[src];
            i += 1;
            src += 1;
        }
    }
    while (src < s.len) {
        result[i] = s[src];
        i += 1;
        src += 1;
    }
    return result;
}

fn ascii_caseless_eql(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |ca, cb| {
        if (std.ascii.toLower(ca) != std.ascii.toLower(cb)) return false;
    }
    return true;
}

pub const Plot = struct {
    allocator: std.mem.Allocator,
    traces: std.ArrayList(Trace),
    strings: std.ArrayList([]u8),
    adopted: std.ArrayList([]align(8) u8),
    layout: Layout = .{},

    pub fn init(allocator: std.mem.Allocator) error{OutOfMemory}!Plot {
        return .{
            .allocator = allocator,
            .traces = std.ArrayList(Trace).empty,
            .strings = std.ArrayList([]u8).empty,
            .adopted = std.ArrayList([]align(8) u8).empty,
        };
    }

    pub fn deinit(self: *Plot) void {
        for (self.strings.items) |s| {
            self.allocator.free(s);
        }
        self.strings.deinit(self.allocator);
        for (self.adopted.items) |s| {
            self.allocator.free(s);
        }
        self.adopted.deinit(self.allocator);
        self.traces.deinit(self.allocator);
    }

    pub fn store_string(self: *Plot, s: []const u8) error{OutOfMemory}![]u8 {
        const duped = try self.allocator.dupe(u8, s);
        errdefer self.allocator.free(duped);
        try self.strings.append(self.allocator, duped);
        return duped;
    }

    /// Take ownership of `bytes` and store it in the plot's adopted byte buffer.
    /// Returns the stable slice; caller must not free it separately.
    pub fn adopt_bytes(self: *Plot, bytes: []align(8) u8) error{OutOfMemory}![]align(8) u8 {
        try self.adopted.append(self.allocator, bytes);
        return bytes;
    }

    pub fn scatter(self: *Plot, t: ScatterTrace) error{OutOfMemory}!void {
        try self.traces.append(self.allocator, Trace{ .scatter = t });
    }

    pub fn line(self: *Plot, t: LineTrace) error{OutOfMemory}!void {
        try self.traces.append(self.allocator, Trace{ .line = t });
    }

    pub fn bar(self: *Plot, t: BarTrace) error{OutOfMemory}!void {
        try self.traces.append(self.allocator, Trace{ .bar = t });
    }

    pub fn heatmap(self: *Plot, t: HeatmapTrace) error{OutOfMemory}!void {
        try self.traces.append(self.allocator, Trace{ .heatmap = t });
    }

    pub fn set_layout(self: *Plot, l: Layout) void {
        self.layout = l;
    }

    fn traces_json(self: Plot) error{ OutOfMemory, InvalidDimension }![]u8 {
        var aw: std.Io.Writer.Allocating = .init(self.allocator);
        errdefer aw.deinit();
        var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

        jw.beginArray() catch |err| switch (err) {
            error.WriteFailed => return error.OutOfMemory,
        };
        for (self.traces.items) |trace| {
            trace.write_json(&jw) catch |err| switch (err) {
                error.WriteFailed => return error.OutOfMemory,
                error.InvalidDimension => return error.InvalidDimension,
            };
        }
        jw.endArray() catch |err| switch (err) {
            error.WriteFailed => return error.OutOfMemory,
        };

        return try aw.toOwnedSlice();
    }

    fn layout_json(self: Plot) error{OutOfMemory}![]u8 {
        var aw: std.Io.Writer.Allocating = .init(self.allocator);
        errdefer aw.deinit();
        var jw = std.json.Stringify{ .writer = &aw.writer, .options = .{ .whitespace = .minified } };

        self.layout.write_json(&jw) catch |err| switch (err) {
            error.WriteFailed => return error.OutOfMemory,
        };

        return try aw.toOwnedSlice();
    }

    pub fn to_html(self: *Plot) error{ OutOfMemory, InvalidDimension }![]u8 {
        const t_json = try self.traces_json();
        defer self.allocator.free(t_json);
        const l_json = try self.layout_json();
        defer self.allocator.free(l_json);

        const t_json_escaped = try escape_script_close(self.allocator, t_json);
        defer self.allocator.free(t_json_escaped);
        const l_json_escaped = try escape_script_close(self.allocator, l_json);
        defer self.allocator.free(l_json_escaped);

        const title = if (self.layout.title.len > 0) self.layout.title else "zsl plot";
        const title_escaped = try escape_html(self.allocator, title);
        defer self.allocator.free(title_escaped);

        return try std.fmt.allocPrint(self.allocator, html_template, .{
            title_escaped,
            self.layout.width,
            self.layout.height,
            t_json_escaped,
            l_json_escaped,
        });
    }
};

test "Plot.store_string stores and frees strings" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    const s = try plot.store_string("hello");
    try std.testing.expect(std.mem.eql(u8, s, "hello"));
}

test "Plot.to_html contains Plotly script and title" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.scatter(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
    });
    plot.set_layout(.{
        .title = "Test",
        .width = 400,
        .height = 300,
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "https://cdn.plot.ly/plotly-2.26.2.min.js") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "Plotly.newPlot") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "<title>Test</title>") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "width:400px") != null);
}

test "Plot.to_html returns InvalidDimension for mismatched scatter trace" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.scatter(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5 },
    });

    try std.testing.expectError(error.InvalidDimension, plot.to_html());
}

test "Plot.to_html escapes title special characters" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    plot.set_layout(.{
        .title = "A <B> & \"C\"",
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "<title>A &lt;B&gt; &amp; &quot;C&quot;</title>") != null);
}

test "Plot.to_html escapes </script> in trace name" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.scatter(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
        .name = "</script><script>alert('x')</script>",
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    // The trace name's </script> sequences must be escaped in the data JSON.
    try std.testing.expect(std.mem.indexOf(u8, html, "\"name\":\"<\\/script><script>alert('x')<\\/script>\"") != null);
}

test "Plot.to_html escapes uppercase </SCRIPT> in trace name" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.scatter(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
        .name = "</SCRIPT>",
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "\"name\":\"<\\/SCRIPT>\"") != null);
}

test "Plot.to_html escapes </Script > with whitespace before >" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.scatter(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
        .name = "</Script >",
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "\"name\":\"<\\/Script >\"") != null);
}

test "Plot.to_html escapes </script> in layout title" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    plot.set_layout(.{
        .title = "</script>",
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "\"title\":\"<\\/script>\"") != null);
}

test "Plot.to_html works with heatmap trace" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.heatmap(.{
        .z = &.{
            &.{ 1, 2 },
            &.{ 3, 4 },
        },
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "\"type\":\"heatmap\"") != null);
}

test "Plot.to_html works with bar trace" {
    const allocator = std.testing.allocator;
    var plot = try Plot.init(allocator);
    defer plot.deinit();

    try plot.bar(.{
        .x = &.{ 1, 2, 3 },
        .y = &.{ 4, 5, 6 },
    });

    const html = try plot.to_html();
    defer allocator.free(html);

    try std.testing.expect(std.mem.indexOf(u8, html, "\"type\":\"bar\"") != null);
}
