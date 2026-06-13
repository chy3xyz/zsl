const std = @import("std");
const la = @import("../la.zig");
const Error = @import("../errors.zig").Error;
const Matrix = la.Matrix;

pub const CsvReadConfig = struct {
    delimiter: u8 = ',',
    skip_header: bool = false,
    skip_rows: usize = 0,
    use_cols: []const usize = &[_]usize{},
    max_rows: i64 = -1,
    comment: u8 = '#',
};

pub const CsvWriteConfig = struct {
    delimiter: u8 = ',',
    header: []const []const u8 = &[_][]const u8{},
};

pub const CsvData = struct {
    data: []const []const f64,
    header: []const []const u8,
    n_rows: usize,
    n_cols: usize,
    raw_strings: []const []const []const u8,
};

pub fn deinit_csv_data(cdata: CsvData, allocator: std.mem.Allocator) void {
    for (cdata.data) |row| allocator.free(@constCast(row));
    allocator.free(@constCast(cdata.data));
    for (cdata.raw_strings) |row| {
        for (row) |s| allocator.free(@constCast(s));
        allocator.free(@constCast(row));
    }
    allocator.free(@constCast(cdata.raw_strings));
    for (cdata.header) |h| allocator.free(@constCast(h));
    allocator.free(@constCast(cdata.header));
}

fn dupe_z(allocator: std.mem.Allocator, s: []const u8) Error![:0]u8 {
    const buf = try allocator.alloc(u8, s.len + 1);
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

fn read_file_to_string(path: []const u8, allocator: std.mem.Allocator) Error![]const u8 {
    const c_path = try dupe_z(allocator, path);
    defer allocator.free(c_path);
    const file = std.c.fopen(c_path, "rb") orelse return error.FileReadError;
    defer _ = std.c.fclose(file);
    var result: std.ArrayList(u8) = .empty;
    defer result.deinit(allocator);
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = std.c.fread(&chunk, 1, chunk.len, file);
        if (n == 0) break;
        try result.appendSlice(allocator, chunk[0..n]);
    }
    return result.toOwnedSlice(allocator);
}

fn split_string(allocator: std.mem.Allocator, line: []const u8, delimiter: u8) Error![]const []const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    defer parts.deinit(allocator);
    var it = std.mem.splitScalar(u8, line, delimiter);
    while (it.next()) |part| {
        try parts.append(allocator, std.mem.trim(u8, part, &std.ascii.whitespace));
    }
    return parts.toOwnedSlice(allocator);
}

pub fn read_csv(path: []const u8, config: CsvReadConfig, allocator: std.mem.Allocator) Error!CsvData {
    const content = try read_file_to_string(path, allocator);
    defer allocator.free(content);
    return try parse_csv(content, config, allocator);
}

pub fn parse_csv(content: []const u8, config: CsvReadConfig, allocator: std.mem.Allocator) Error!CsvData {
    var data_rows: std.ArrayList([]f64) = .empty;
    defer {
        for (data_rows.items) |row| allocator.free(row);
        data_rows.deinit(allocator);
    }
    var raw_rows: std.ArrayList([]const []const u8) = .empty;
    defer {
        for (raw_rows.items) |row| allocator.free(@constCast(row));
        raw_rows.deinit(allocator);
    }
    var header: std.ArrayList([]const u8) = .empty;
    defer {
        for (header.items) |h| allocator.free(h);
        header.deinit(allocator);
    }

    var skip_count = config.skip_rows;
    var header_parsed = false;
    var row_count: i64 = 0;

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, &std.ascii.whitespace);
        if (trimmed.len == 0) continue;
        if (trimmed[0] == config.comment) continue;
        if (skip_count > 0) {
            skip_count -= 1;
            continue;
        }
        if (config.skip_header and !header_parsed) {
            const parts = try split_string(allocator, trimmed, config.delimiter);
            defer allocator.free(parts);
            for (parts) |p| {
                const trimmed_part = std.mem.trim(u8, p, "\"'");
                try header.append(allocator, try allocator.dupe(u8, trimmed_part));
            }
            header_parsed = true;
            continue;
        }
        if (config.max_rows >= 0 and row_count >= config.max_rows) break;

        const parts = try split_string(allocator, trimmed, config.delimiter);
        defer allocator.free(parts);
        var row: std.ArrayList(f64) = .empty;
        defer row.deinit(allocator);
        var raw_row: std.ArrayList([]const u8) = .empty;
        defer {
            for (raw_row.items) |s| allocator.free(s);
            raw_row.deinit(allocator);
        }

        for (parts, 0..) |field, j| {
            if (config.use_cols.len > 0) {
                var use = false;
                for (config.use_cols) |col| {
                    if (col == j) {
                        use = true;
                        break;
                    }
                }
                if (!use) continue;
            }
            const clean = std.mem.trim(u8, field, "\"'");
            const val = std.fmt.parseFloat(f64, clean) catch return error.ParseError;
            try row.append(allocator, val);
            try raw_row.append(allocator, try allocator.dupe(u8, clean));
        }
        if (row.items.len > 0) {
            try data_rows.append(allocator, try row.toOwnedSlice(allocator));
            try raw_rows.append(allocator, try raw_row.toOwnedSlice(allocator));
            row_count += 1;
        }
    }

    const data = try data_rows.toOwnedSlice(allocator);
    const raw_strings = try raw_rows.toOwnedSlice(allocator);
    const n_cols = if (data.len > 0) data[0].len else 0;
    return CsvData{
        .data = data,
        .header = try header.toOwnedSlice(allocator),
        .n_rows = data.len,
        .n_cols = n_cols,
        .raw_strings = raw_strings,
    };
}

pub fn read_csv_to_matrix(path: []const u8, config: CsvReadConfig, allocator: std.mem.Allocator) Error!Matrix(f64) {
    const csv_data = try read_csv(path, config, allocator);
    defer deinit_csv_data(csv_data, allocator);
    if (csv_data.n_rows == 0 or csv_data.n_cols == 0) return error.InvalidDimension;
    var mat = try Matrix(f64).init(allocator, csv_data.n_rows, csv_data.n_cols);
    errdefer mat.deinit(allocator);
    for (0..csv_data.n_rows) |i| {
        for (0..csv_data.n_cols) |j| {
            try mat.set(i, j, csv_data.data[i][j]);
        }
    }
    return mat;
}

fn format_float(allocator: std.mem.Allocator, val: f64) Error![]const u8 {
    if (std.math.isNan(val)) return try allocator.dupe(u8, "NaN");
    return try std.fmt.allocPrint(allocator, "{d:.6}", .{val});
}

pub fn write_csv(path: []const u8, data: []const []const f64, config: CsvWriteConfig, allocator: std.mem.Allocator) Error!void {
    var lines: std.ArrayList([]const u8) = .empty;
    defer {
        for (lines.items) |line| allocator.free(line);
        lines.deinit(allocator);
    }
    if (config.header.len > 0) {
        const header_line = try std.mem.join(allocator, &[_]u8{config.delimiter}, config.header);
        try lines.append(allocator, header_line);
    }
    for (data) |row| {
        var fields: std.ArrayList([]const u8) = .empty;
        defer {
            for (fields.items) |f| allocator.free(f);
            fields.deinit(allocator);
        }
        for (row) |val| {
            try fields.append(allocator, try format_float(allocator, val));
        }
        const row_line = try std.mem.join(allocator, &[_]u8{config.delimiter}, fields.items);
        try lines.append(allocator, row_line);
    }
    const out = try std.mem.join(allocator, "\n", lines.items);
    defer allocator.free(out);

    const c_path = try dupe_z(allocator, path);
    defer allocator.free(c_path);
    const file = std.c.fopen(c_path, "wb") orelse return error.FileWriteError;
    defer _ = std.c.fclose(file);
    _ = std.c.fwrite(out.ptr, 1, out.len, file);
}

pub fn write_matrix_csv(path: []const u8, mat: Matrix(f64), config: CsvWriteConfig, allocator: std.mem.Allocator) Error!void {
    var data: std.ArrayList([]const f64) = .empty;
    defer {
        for (data.items) |row| allocator.free(row);
        data.deinit(allocator);
    }
    for (0..mat.rows) |i| {
        var row: std.ArrayList(f64) = .empty;
        defer row.deinit(allocator);
        for (0..mat.cols) |j| {
            try row.append(allocator, try mat.get(i, j));
        }
        try data.append(allocator, try row.toOwnedSlice(allocator));
    }
    return try write_csv(path, data.items, config, allocator);
}

pub fn to_matrix(csv_data: CsvData, allocator: std.mem.Allocator) Error!Matrix(f64) {
    if (csv_data.n_rows == 0 or csv_data.n_cols == 0) return error.InvalidDimension;
    var mat = try Matrix(f64).init(allocator, csv_data.n_rows, csv_data.n_cols);
    errdefer mat.deinit(allocator);
    for (0..csv_data.n_rows) |i| {
        for (0..csv_data.n_cols) |j| {
            try mat.set(i, j, csv_data.data[i][j]);
        }
    }
    return mat;
}

pub fn get_column(csv_data: CsvData, idx: usize, allocator: std.mem.Allocator) Error![]f64 {
    if (idx >= csv_data.n_cols) return error.IndexOutOfBounds;
    var col = try allocator.alloc(f64, csv_data.n_rows);
    for (0..csv_data.n_rows) |i| {
        col[i] = csv_data.data[i][idx];
    }
    return col;
}

pub fn get_column_by_name(csv_data: CsvData, name: []const u8, allocator: std.mem.Allocator) Error![]f64 {
    for (csv_data.header, 0..) |h, i| {
        if (std.mem.eql(u8, h, name)) {
            return try get_column(csv_data, i, allocator);
        }
    }
    return error.IndexOutOfBounds;
}

test "csv round trip" {
    const allocator = std.testing.allocator;
    const T = f64;
    const M = Matrix(T);
    var mat = try M.fromRowSlice(allocator, 2, 3, &[_]T{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    });
    defer mat.deinit(allocator);

    const path = "/tmp/zsl_csv_test.csv";
    try write_matrix_csv(path, mat, .{}, allocator);

    var read_mat = try read_csv_to_matrix(path, .{}, allocator);
    defer read_mat.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), read_mat.rows);
    try std.testing.expectEqual(@as(usize, 3), read_mat.cols);
    for (0..2) |i| {
        for (0..3) |j| {
            try std.testing.expectEqual(try mat.get(i, j), try read_mat.get(i, j));
        }
    }
}

test "csv parse with header and column access" {
    const allocator = std.testing.allocator;
    const csv_text =
        \\x,y,z
        \\1.0,2.0,3.0
        \\4.0,5.0,6.0
    ;
    const csv_data = try parse_csv(csv_text, .{ .skip_header = true }, allocator);
    defer deinit_csv_data(csv_data, allocator);

    try std.testing.expectEqual(@as(usize, 2), csv_data.n_rows);
    try std.testing.expectEqual(@as(usize, 3), csv_data.n_cols);
    try std.testing.expectEqualStrings("x", csv_data.header[0]);
    try std.testing.expectEqualStrings("y", csv_data.header[1]);
    try std.testing.expectEqualStrings("z", csv_data.header[2]);

    const ys = try get_column_by_name(csv_data, "y", allocator);
    defer allocator.free(ys);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), ys[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), ys[1], 1e-9);

    const zs = try get_column(csv_data, 2, allocator);
    defer allocator.free(zs);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), zs[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), zs[1], 1e-9);
}

test "csv read missing file returns FileReadError" {
    const allocator = std.testing.allocator;
    const result = read_csv("/tmp/zsl_nonexistent_file.csv", .{}, allocator);
    try std.testing.expectError(error.FileReadError, result);
}

test "csv parse invalid numeric returns ParseError" {
    const allocator = std.testing.allocator;
    const csv_text = "1.0,not_a_number\n";
    const result = parse_csv(csv_text, .{}, allocator);
    try std.testing.expectError(error.ParseError, result);
}

test "csv get_column_by_name missing returns IndexOutOfBounds" {
    const allocator = std.testing.allocator;
    const csv_text = "x,y\n1.0,2.0\n";
    const csv_data = try parse_csv(csv_text, .{ .skip_header = true }, allocator);
    defer deinit_csv_data(csv_data, allocator);

    const result = get_column_by_name(csv_data, "z", allocator);
    try std.testing.expectError(error.IndexOutOfBounds, result);
}
