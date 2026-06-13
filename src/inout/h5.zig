const std = @import("std");
const Error = @import("../errors.zig").Error;

/// Error set for HDF5 operations.  Includes the shared `Error` variants plus
/// HDF5-specific open/close errors.
pub const H5Error = Error || error{
    FileOpenError,
    FileCloseError,
};

const root = @import("root");

/// `true` when the coordinator (or root module) has enabled the real HDF5 C binding.
/// When `false` (the default), all file operations return `error.NotImplemented`.
///
/// To enable real I/O the coordinator must:
///   1. Add `-Dhdf5` (or a build option) and call `linkSystemLibrary("hdf5")` in `build.zig`.
///   2. Expose `pub const hdf5_enabled = true;` from the root module so this file picks it up.
pub const hdf5_enabled = if (@hasDecl(root, "hdf5_enabled")) root.hdf5_enabled else false;

/// File open/create mode.
pub const FileMode = enum {
    /// Open an existing file read-only.
    read,
    /// Open an existing file read-write.
    write,
    /// Create a new file, truncating any existing file.
    truncate,
};

/// A handle to an HDF5 file.
pub const H5File = struct {
    /// HDF5 file identifier (valid only in real mode).
    handle: i64 = 0,

    /// Open or create an HDF5 file.
    pub fn open(path: []const u8, mode: FileMode) H5Error!H5File {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_path = try dupeZ(std.heap.page_allocator, path);
        defer std.heap.page_allocator.free(c_path);

        const flags: c_uint = switch (mode) {
            .read => H5F_ACC_RDONLY,
            .write => H5F_ACC_RDWR,
            .truncate => H5F_ACC_TRUNC,
        };

        const file_id = if (mode == .truncate)
            c.H5Fcreate(c_path, flags, H5P_DEFAULT, H5P_DEFAULT)
        else
            c.H5Fopen(c_path, flags, H5P_DEFAULT);

        if (file_id < 0) return error.FileOpenError;
        return .{ .handle = file_id };
    }

    /// Close the HDF5 file.
    pub fn close(self: *H5File) H5Error!void {
        if (!hdf5_enabled) return error.NotImplemented;
        if (c.H5Fclose(self.handle) < 0) return error.FileCloseError;
        self.handle = 0;
    }

    /// Create a 1-D `f64` dataset named `name`.
    pub fn create_dataset_1d(self: *H5File, name: []const u8, data: []const f64) H5Error!void {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_name = try dupeZ(std.heap.page_allocator, name);
        defer std.heap.page_allocator.free(c_name);

        const dims: [1]c.hsize_t = .{data.len};
        const err = c.H5LTmake_dataset(
            self.handle,
            c_name,
            1,
            &dims,
            c.H5T_IEEE_F64LE_g,
            data.ptr,
        );
        if (err < 0) return error.FileWriteError;
    }

    /// Create a 2-D `f64` dataset named `name`.  All rows must have the same length.
    pub fn create_dataset_2d(self: *H5File, name: []const u8, data: []const []const f64) H5Error!void {
        if (!hdf5_enabled) return error.NotImplemented;
        if (data.len == 0) return error.InvalidDimension;
        const rows = data.len;
        const cols = data[0].len;
        if (cols == 0) return error.InvalidDimension;

        const flat = try std.heap.page_allocator.alloc(f64, rows * cols);
        defer std.heap.page_allocator.free(flat);

        for (data, 0..) |row, i| {
            if (row.len != cols) return error.InvalidDimension;
            @memcpy(flat[i * cols ..][0..cols], row);
        }

        const c_name = try dupeZ(std.heap.page_allocator, name);
        defer std.heap.page_allocator.free(c_name);

        const dims: [2]c.hsize_t = .{ rows, cols };
        const err = c.H5LTmake_dataset(
            self.handle,
            c_name,
            2,
            &dims,
            c.H5T_IEEE_F64LE_g,
            flat.ptr,
        );
        if (err < 0) return error.FileWriteError;
    }

    /// Create a 3-D `f64` dataset named `name`.  All sub-arrays must have consistent dimensions.
    pub fn create_dataset_3d(self: *H5File, name: []const u8, data: []const []const []const f64) H5Error!void {
        if (!hdf5_enabled) return error.NotImplemented;
        if (data.len == 0) return error.InvalidDimension;
        const d1 = data.len;
        const d2 = data[0].len;
        const d3 = if (d2 > 0) data[0][0].len else 0;
        if (d2 == 0 or d3 == 0) return error.InvalidDimension;

        const flat = try std.heap.page_allocator.alloc(f64, d1 * d2 * d3);
        defer std.heap.page_allocator.free(flat);

        for (data, 0..) |layer, i| {
            if (layer.len != d2) return error.InvalidDimension;
            for (layer, 0..) |row, j| {
                if (row.len != d3) return error.InvalidDimension;
                const offset = (i * d2 + j) * d3;
                @memcpy(flat[offset..][0..d3], row);
            }
        }

        const c_name = try dupeZ(std.heap.page_allocator, name);
        defer std.heap.page_allocator.free(c_name);

        const dims: [3]c.hsize_t = .{ d1, d2, d3 };
        const err = c.H5LTmake_dataset(
            self.handle,
            c_name,
            3,
            &dims,
            c.H5T_IEEE_F64LE_g,
            flat.ptr,
        );
        if (err < 0) return error.FileWriteError;
    }

    /// Read a 1-D `f64` dataset.  Caller owns the returned slice and must free it with `allocator`.
    pub fn read_dataset_1d(self: *H5File, allocator: std.mem.Allocator, name: []const u8) H5Error![]f64 {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_name = try dupeZ(std.heap.page_allocator, name);
        defer std.heap.page_allocator.free(c_name);

        var rank: c_int = 0;
        if (c.H5LTget_dataset_ndims(self.handle, c_name, &rank) < 0) return error.FileReadError;
        if (rank != 1) return error.InvalidDimension;

        var dims: [1]c.hsize_t = undefined;
        var class_id: c_int = 0;
        var type_size: usize = 0;
        if (c.H5LTget_dataset_info(self.handle, c_name, &dims[0], &class_id, &type_size) < 0) {
            return error.FileReadError;
        }

        const result = try allocator.alloc(f64, dims[0]);
        errdefer allocator.free(result);

        if (c.H5LTread_dataset(self.handle, c_name, c.H5T_NATIVE_DOUBLE_g, result.ptr) < 0) {
            return error.FileReadError;
        }
        return result;
    }

    /// A 2-D dataset returned by `read_dataset_2d`.
    pub const Dataset2d = struct {
        rows: usize,
        cols: usize,
        data: []f64,
    };

    /// Read a 2-D `f64` dataset.  Caller owns `Dataset2d.data` and must free it with `allocator`.
    pub fn read_dataset_2d(self: *H5File, allocator: std.mem.Allocator, name: []const u8) H5Error!Dataset2d {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_name = try dupeZ(std.heap.page_allocator, name);
        defer std.heap.page_allocator.free(c_name);

        var rank: c_int = 0;
        if (c.H5LTget_dataset_ndims(self.handle, c_name, &rank) < 0) return error.FileReadError;
        if (rank != 2) return error.InvalidDimension;

        var dims: [2]c.hsize_t = undefined;
        var class_id: c_int = 0;
        var type_size: usize = 0;
        if (c.H5LTget_dataset_info(self.handle, c_name, &dims[0], &class_id, &type_size) < 0) {
            return error.FileReadError;
        }
        const rows = dims[0];
        const cols = dims[1];

        const flat = try allocator.alloc(f64, rows * cols);
        errdefer allocator.free(flat);

        if (c.H5LTread_dataset(self.handle, c_name, c.H5T_NATIVE_DOUBLE_g, flat.ptr) < 0) {
            return error.FileReadError;
        }
        return .{ .rows = rows, .cols = cols, .data = flat };
    }

    /// A 3-D dataset returned by `read_dataset_3d`.
    pub const Dataset3d = struct {
        d1: usize,
        d2: usize,
        d3: usize,
        data: []f64,
    };

    /// Read a 3-D `f64` dataset.  Caller owns `Dataset3d.data` and must free it with `allocator`.
    pub fn read_dataset_3d(self: *H5File, allocator: std.mem.Allocator, name: []const u8) H5Error!Dataset3d {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_name = try dupeZ(std.heap.page_allocator, name);
        defer std.heap.page_allocator.free(c_name);

        var rank: c_int = 0;
        if (c.H5LTget_dataset_ndims(self.handle, c_name, &rank) < 0) return error.FileReadError;
        if (rank != 3) return error.InvalidDimension;

        var dims: [3]c.hsize_t = undefined;
        var class_id: c_int = 0;
        var type_size: usize = 0;
        if (c.H5LTget_dataset_info(self.handle, c_name, &dims[0], &class_id, &type_size) < 0) {
            return error.FileReadError;
        }
        const d1 = dims[0];
        const d2 = dims[1];
        const d3 = dims[2];

        const flat = try allocator.alloc(f64, d1 * d2 * d3);
        errdefer allocator.free(flat);

        if (c.H5LTread_dataset(self.handle, c_name, c.H5T_NATIVE_DOUBLE_g, flat.ptr) < 0) {
            return error.FileReadError;
        }
        return .{ .d1 = d1, .d2 = d2, .d3 = d3, .data = flat };
    }

    /// Write a scalar `f64` attribute on a dataset.
    pub fn write_attribute(self: *H5File, dset_name: []const u8, attr_name: []const u8, value: f64) H5Error!void {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_dset = try dupeZ(std.heap.page_allocator, dset_name);
        defer std.heap.page_allocator.free(c_dset);
        const c_attr = try dupeZ(std.heap.page_allocator, attr_name);
        defer std.heap.page_allocator.free(c_attr);

        const err = c.H5LTset_attribute_double(self.handle, c_dset, c_attr, &value, 1);
        if (err < 0) return error.FileWriteError;
    }

    /// Read a scalar `f64` attribute from a dataset.
    pub fn read_attribute(self: *H5File, dset_name: []const u8, attr_name: []const u8) H5Error!f64 {
        if (!hdf5_enabled) return error.NotImplemented;
        const c_dset = try dupeZ(std.heap.page_allocator, dset_name);
        defer std.heap.page_allocator.free(c_dset);
        const c_attr = try dupeZ(std.heap.page_allocator, attr_name);
        defer std.heap.page_allocator.free(c_attr);

        var value: f64 = 0.0;
        if (c.H5LTget_attribute_double(self.handle, c_dset, c_attr, &value) < 0) {
            return error.FileReadError;
        }
        return value;
    }
};

// ---------------------------------------------------------------------------
// C binding declarations (only analyzed when hdf5_enabled is true).
// ---------------------------------------------------------------------------
const c = if (hdf5_enabled) struct {
    const hid_t = i64;
    const herr_t = c_int;
    const hsize_t = u64;

    extern "c" fn H5Fcreate([*c]const u8, c_uint, hid_t, hid_t) hid_t;
    extern "c" fn H5Fopen([*c]const u8, c_uint, hid_t) hid_t;
    extern "c" fn H5Fclose(hid_t) herr_t;

    extern "c" fn H5LTmake_dataset(hid_t, [*c]const u8, c_int, [*c]const hsize_t, hid_t, ?*const anyopaque) herr_t;
    extern "c" fn H5LTread_dataset(hid_t, [*c]const u8, hid_t, ?*anyopaque) herr_t;
    extern "c" fn H5LTget_dataset_ndims(hid_t, [*c]const u8, *c_int) herr_t;
    extern "c" fn H5LTget_dataset_info(hid_t, [*c]const u8, *hsize_t, *c_int, *usize) herr_t;

    extern "c" fn H5LTset_attribute_double(hid_t, [*c]const u8, [*c]const u8, *const f64, u64) herr_t;
    extern "c" fn H5LTget_attribute_double(hid_t, [*c]const u8, [*c]const u8, *f64) herr_t;

    extern "c" var H5T_IEEE_F64LE_g: hid_t;
    extern "c" var H5T_NATIVE_DOUBLE_g: hid_t;
} else struct {};

const H5F_ACC_RDONLY: c_uint = 0x0000;
const H5F_ACC_RDWR: c_uint = 0x0001;
const H5F_ACC_TRUNC: c_uint = 0x0002;
const H5P_DEFAULT: c.hid_t = 0;

fn dupeZ(allocator: std.mem.Allocator, s: []const u8) Error![:0]u8 {
    const buf = try allocator.alloc(u8, s.len + 1);
    errdefer allocator.free(buf);
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
test "h5 stub returns NotImplemented" {
    if (hdf5_enabled) return error.SkipZigTest;

    try std.testing.expectError(error.NotImplemented, H5File.open("test.h5", .read));

    var file = H5File{ .handle = 0 };
    try std.testing.expectError(error.NotImplemented, file.create_dataset_1d("x", &[_]f64{ 1.0, 2.0 }));
    try std.testing.expectError(error.NotImplemented, file.create_dataset_2d("m", &[_][]const f64{ &[_]f64{1.0}, &[_]f64{2.0} }));
    try std.testing.expectError(error.NotImplemented, file.create_dataset_3d("t", &[_][]const []const f64{&[_][]const f64{&[_]f64{1.0}}}));
    try std.testing.expectError(error.NotImplemented, file.read_dataset_1d(std.testing.allocator, "x"));
    try std.testing.expectError(error.NotImplemented, file.read_dataset_2d(std.testing.allocator, "m"));
    try std.testing.expectError(error.NotImplemented, file.read_dataset_3d(std.testing.allocator, "t"));
    try std.testing.expectError(error.NotImplemented, file.write_attribute("x", "scale", 1.0));
    try std.testing.expectError(error.NotImplemented, file.read_attribute("x", "scale"));
    try std.testing.expectError(error.NotImplemented, file.close());
}

test "h5 public types are defined" {
    _ = FileMode.read;
    _ = FileMode.write;
    _ = FileMode.truncate;
    _ = H5File{};
    _ = H5File.Dataset2d{ .rows = 0, .cols = 0, .data = &[_]f64{} };
    _ = H5File.Dataset3d{ .d1 = 0, .d2 = 0, .d3 = 0, .data = &[_]f64{} };
}
