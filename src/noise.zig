const std = @import("std");

/// Classic Perlin permutation table (256 entries). The lookup table is
/// duplicated internally to 512 entries so that index calculations never
/// overflow.
const permutation = [256]u8{
    151, 160, 137, 91,  90,  15,  131, 13,  201, 95,  96,  53,  194, 233, 7,   225,
    140, 36,  103, 30,  69,  142, 8,   99,  37,  240, 21,  10,  23,  190, 6,   148,
    247, 120, 234, 75,  0,   26,  197, 62,  94,  252, 219, 203, 117, 35,  11,  32,
    57,  177, 33,  88,  237, 149, 56,  87,  174, 20,  125, 136, 171, 168, 68,  175,
    74,  165, 71,  134, 139, 48,  27,  166, 77,  146, 158, 231, 83,  111, 229, 122,
    60,  211, 133, 230, 220, 105, 92,  41,  55,  46,  245, 40,  244, 102, 143, 54,
    65,  25,  63,  161, 1,   216, 80,  73,  209, 76,  132, 187, 208, 89,  18,  169,
    200, 196, 135, 130, 116, 188, 159, 86,  164, 100, 109, 198, 173, 186, 3,   64,
    52,  217, 226, 250, 124, 123, 5,   202, 38,  147, 118, 126, 255, 82,  85,  212,
    207, 206, 59,  227, 47,  16,  58,  17,  182, 189, 28,  42,  223, 183, 170, 213,
    119, 248, 152, 2,   44,  154, 163, 70,  221, 153, 101, 155, 167, 43,  172, 9,
    129, 22,  39,  253, 19,  98,  108, 110, 79,  113, 224, 232, 178, 185, 112, 104,
    218, 246, 97,  228, 251, 34,  242, 193, 238, 210, 144, 12,  191, 179, 162, 241,
    81,  51,  145, 235, 249, 14,  239, 107, 49,  192, 214, 31,  181, 199, 106, 157,
    184, 84,  204, 176, 115, 121, 50,  45,  127, 4,   150, 254, 138, 236, 205, 93,
    222, 114, 67,  29,  24,  72,  243, 141, 128, 195, 78,  66,  215, 61,  156, 180,
};

pub const Generator = struct {
    /// 512-entry permutation table (the classic 256-entry table duplicated).
    perm: [512]u8,

    /// Create a generator with the unshuffled classic permutation table.
    pub fn init_default() Generator {
        return .{ .perm = permutation ++ permutation };
    }

    /// Create a generator whose permutation table is deterministically
    /// shuffled from `seed` using the standard PRNG.
    pub fn init(seed: u64) Generator {
        var perm: [256]u8 = permutation;

        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        rng.shuffle(u8, &perm);

        return .{ .perm = perm ++ perm };
    }
};

/// 2D Perlin noise. Returns a value in approximately [-1, 1].
pub fn perlin2d(g: Generator, x: f64, y: f64) f64 {
    const ix = @as(i32, @intFromFloat(x));
    const iy = @as(i32, @intFromFloat(y));

    const xi: usize = @intCast(ix & 0xFF);
    const yi: usize = @intCast(iy & 0xFF);

    const xf = x - @as(f64, @floatFromInt(ix));
    const yf = y - @as(f64, @floatFromInt(iy));

    const u = fade(xf);
    const v = fade(yf);

    const pxi = g.perm[xi];
    const pxi1 = g.perm[xi + 1];

    const aa = g.perm[pxi + yi];
    const ab = g.perm[pxi + yi + 1];
    const ba = g.perm[pxi1 + yi];
    const bb = g.perm[pxi1 + yi + 1];

    const x1 = lerp(grad2d(aa, xf, yf), grad2d(ba, xf - 1, yf), u);
    const x2 = lerp(grad2d(ab, xf, yf - 1), grad2d(bb, xf - 1, yf - 1), u);

    return lerp(x1, x2, v);
}

/// 3D Perlin noise. Returns a value in approximately [-1, 1].
pub fn perlin3d(g: Generator, x: f64, y: f64, z: f64) f64 {
    const ix = @as(i32, @intFromFloat(x));
    const iy = @as(i32, @intFromFloat(y));
    const iz = @as(i32, @intFromFloat(z));

    const xi: usize = @intCast(ix & 0xFF);
    const yi: usize = @intCast(iy & 0xFF);
    const zi: usize = @intCast(iz & 0xFF);

    const xf = x - @as(f64, @floatFromInt(ix));
    const yf = y - @as(f64, @floatFromInt(iy));
    const zf = z - @as(f64, @floatFromInt(iz));

    const u = fade(xf);
    const v = fade(yf);
    const w = fade(zf);

    const pxi = g.perm[xi];
    const pxi_yi = g.perm[pxi + yi];
    const pxi_yi1 = g.perm[pxi + yi + 1];
    const pxi1 = g.perm[xi + 1];
    const pxi1_yi = g.perm[pxi1 + yi];
    const pxi1_yi1 = g.perm[pxi1 + yi + 1];

    const aaa = g.perm[pxi_yi + zi];
    const aba = g.perm[pxi_yi1 + zi];
    const aab = g.perm[pxi_yi + zi + 1];
    const abb = g.perm[pxi_yi1 + zi + 1];
    const baa = g.perm[pxi1_yi + zi];
    const bba = g.perm[pxi1_yi1 + zi];
    const bab = g.perm[pxi1_yi + zi + 1];
    const bbb = g.perm[pxi1_yi1 + zi + 1];

    const x1 = lerp(grad3d(aaa, xf, yf, zf), grad3d(baa, xf - 1, yf, zf), u);
    const x2 = lerp(grad3d(aba, xf, yf - 1, zf), grad3d(bba, xf - 1, yf - 1, zf), u);
    const y1 = lerp(x1, x2, v);

    const x3 = lerp(grad3d(aab, xf, yf, zf - 1), grad3d(bab, xf - 1, yf, zf - 1), u);
    const x4 = lerp(grad3d(abb, xf, yf - 1, zf - 1), grad3d(bbb, xf - 1, yf - 1, zf - 1), u);
    const y2 = lerp(x3, x4, v);

    return lerp(y1, y2, w);
}

fn fade(t: f64) f64 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f64, b: f64, x: f64) f64 {
    return a + x * (b - a);
}

fn grad2d(hash: u8, x: f64, y: f64) f64 {
    const h: u4 = @intCast(hash & 0xF);
    return switch (h) {
        0x0 => x + y,
        0x1 => -x + y,
        0x2 => x - y,
        0x3 => -x - y,
        0x4 => x,
        0x5 => -x,
        0x6 => x,
        0x7 => -x,
        0x8 => y,
        0x9 => -y,
        0xA => y,
        0xB => -y,
        0xC => y + x,
        0xD => -y,
        0xE => y - x,
        0xF => -y,
    };
}

fn grad3d(hash: u8, x: f64, y: f64, z: f64) f64 {
    const h: u4 = @intCast(hash & 0xF);
    return switch (h) {
        0x0 => x + y,
        0x1 => -x + y,
        0x2 => x - y,
        0x3 => -x - y,
        0x4 => x + z,
        0x5 => -x + z,
        0x6 => x - z,
        0x7 => -x - z,
        0x8 => y + z,
        0x9 => -y + z,
        0xA => y - z,
        0xB => -y - z,
        0xC => y + x,
        0xD => -y + z,
        0xE => y - x,
        0xF => -y - z,
    };
}

// ---------------------------------------------------------------------------
// Simplex noise
// ---------------------------------------------------------------------------

const f2 = 0.5 * (std.math.sqrt(3.0) - 1.0);
const g2 = (3.0 - std.math.sqrt(3.0)) / 6.0;
const f3 = 1.0 / 3.0;
const g3 = 1.0 / 6.0;

/// 2D simplex noise. Returns a value in approximately [-1, 1].
pub fn simplex2d(g: Generator, x: f64, y: f64) f64 {
    const s = (x + y) * f2;
    const i = @as(i32, @intFromFloat(x + s));
    const j = @as(i32, @intFromFloat(y + s));

    const t = @as(f64, @floatFromInt(i + j)) * g2;
    const x0 = x - (@as(f64, @floatFromInt(i)) - t);
    const y0 = y - (@as(f64, @floatFromInt(j)) - t);

    const i1_: usize = if (x0 > y0) 1 else 0;
    const j1_: usize = if (x0 > y0) 0 else 1;

    const x1 = x0 - @as(f64, @floatFromInt(i1_)) + g2;
    const y1 = y0 - @as(f64, @floatFromInt(j1_)) + g2;
    const x2 = x0 - 1.0 + g2 * 2.0;
    const y2 = y0 - 1.0 + g2 * 2.0;

    const ii: usize = @intCast(i & 0xFF);
    const jj: usize = @intCast(j & 0xFF);

    var t0 = 0.5 - x0 * x0 - y0 * y0;
    const n0 = if (t0 < 0.0) 0.0 else blk: {
        t0 *= t0;
        break :blk t0 * t0 * simplex_grad_2d(g.perm[ii + g.perm[jj]], x0, y0);
    };

    var t1 = 0.5 - x1 * x1 - y1 * y1;
    const n1 = if (t1 < 0.0) 0.0 else blk: {
        t1 *= t1;
        break :blk t1 * t1 * simplex_grad_2d(g.perm[ii + i1_ + g.perm[jj + j1_]], x1, y1);
    };

    var t2 = 0.5 - x2 * x2 - y2 * y2;
    const n2 = if (t2 < 0.0) 0.0 else blk: {
        t2 *= t2;
        break :blk t2 * t2 * simplex_grad_2d(g.perm[ii + 1 + g.perm[jj + 1]], x2, y2);
    };

    return 40.0 * (n0 + n1 + n2);
}

/// 3D simplex noise. Returns a value in approximately [-1, 1].
pub fn simplex3d(g: Generator, x: f64, y: f64, z: f64) f64 {
    const s = (x + y + z) * f3;
    const xs = x + s;
    const ys = y + s;
    const zs = z + s;
    const i = @as(i32, @intFromFloat(xs));
    const j = @as(i32, @intFromFloat(ys));
    const k = @as(i32, @intFromFloat(zs));

    const t = @as(f64, @floatFromInt(i + j + k)) * g3;
    const x0 = x - (@as(f64, @floatFromInt(i)) - t);
    const y0 = y - (@as(f64, @floatFromInt(j)) - t);
    const z0 = z - (@as(f64, @floatFromInt(k)) - t);

    var i1_: usize = 0;
    var j1_: usize = 0;
    var k1_: usize = 0;
    var i2_: usize = 0;
    var j2_: usize = 0;
    var k2_: usize = 0;

    if (x0 >= y0) {
        if (y0 >= z0) {
            i1_ = 1;
            i2_ = 1;
            j2_ = 1;
        } else if (x0 >= z0) {
            i1_ = 1;
            i2_ = 1;
            k2_ = 1;
        } else {
            k1_ = 1;
            i2_ = 1;
            k2_ = 1;
        }
    } else {
        if (y0 < z0) {
            k1_ = 1;
            j2_ = 1;
            k2_ = 1;
        } else if (x0 < z0) {
            j1_ = 1;
            j2_ = 1;
            k2_ = 1;
        } else {
            j1_ = 1;
            i2_ = 1;
            j2_ = 1;
        }
    }

    const x1 = x0 - @as(f64, @floatFromInt(i1_)) + g3;
    const y1 = y0 - @as(f64, @floatFromInt(j1_)) + g3;
    const z1 = z0 - @as(f64, @floatFromInt(k1_)) + g3;
    const x2 = x0 - @as(f64, @floatFromInt(i2_)) + 2.0 * g3;
    const y2 = y0 - @as(f64, @floatFromInt(j2_)) + 2.0 * g3;
    const z2 = z0 - @as(f64, @floatFromInt(k2_)) + 2.0 * g3;
    const x3 = x0 - 1.0 + 3.0 * g3;
    const y3 = y0 - 1.0 + 3.0 * g3;
    const z3 = z0 - 1.0 + 3.0 * g3;

    const ii: usize = @intCast(i & 0xFF);
    const jj: usize = @intCast(j & 0xFF);
    const kk: usize = @intCast(k & 0xFF);

    var t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    const n0 = if (t0 < 0.0) 0.0 else blk: {
        t0 *= t0;
        break :blk t0 * t0 * simplex_grad_3d(g.perm[ii + g.perm[jj + g.perm[kk]]], x0, y0, z0);
    };

    var t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    const n1 = if (t1 < 0.0) 0.0 else blk: {
        t1 *= t1;
        break :blk t1 * t1 * simplex_grad_3d(g.perm[ii + i1_ + g.perm[jj + j1_ + g.perm[kk + k1_]]], x1, y1, z1);
    };

    var t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    const n2 = if (t2 < 0.0) 0.0 else blk: {
        t2 *= t2;
        break :blk t2 * t2 * simplex_grad_3d(g.perm[ii + i2_ + g.perm[jj + j2_ + g.perm[kk + k2_]]], x2, y2, z2);
    };

    var t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    const n3 = if (t3 < 0.0) 0.0 else blk: {
        t3 *= t3;
        break :blk t3 * t3 * simplex_grad_3d(g.perm[ii + 1 + g.perm[jj + 1 + g.perm[kk + 1]]], x3, y3, z3);
    };

    return 32.0 * (n0 + n1 + n2 + n3);
}

fn simplex_grad_2d(hash: u8, x: f64, y: f64) f64 {
    const h = hash & 7;
    const u = if (h < 4) x else y;
    const v = if (h < 4) y else x;
    return (if (h & 1 == 0) u else -u) + (if (h & 2 == 0) 2.0 * v else -2.0 * v);
}

fn simplex_grad_3d(hash: u8, x: f64, y: f64, z: f64) f64 {
    const h = hash & 15;
    const u = if (h < 8) x else y;
    const v = if (h < 4) y else if (h == 12 or h == 14) x else z;
    return (if (h & 1 == 0) u else -u) + (if (h & 2 == 0) v else -v);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "init_default is deterministic" {
    const gen1 = Generator.init_default();
    const gen2 = Generator.init_default();
    try std.testing.expectEqual(gen1.perm, gen2.perm);
}

test "init is reproducible" {
    const gen1 = Generator.init(12_345);
    const gen2 = Generator.init(12_345);
    try std.testing.expectEqual(gen1.perm, gen2.perm);
}

test "same seed produces identical noise values" {
    const gen1 = Generator.init(99_999);
    const gen2 = Generator.init(99_999);

    try std.testing.expectApproxEqAbs(perlin2d(gen1, 1.2, 3.4), perlin2d(gen2, 1.2, 3.4), 1e-12);
    try std.testing.expectApproxEqAbs(perlin3d(gen1, 1.2, 3.4, 5.6), perlin3d(gen2, 1.2, 3.4, 5.6), 1e-12);
    try std.testing.expectApproxEqAbs(simplex2d(gen1, 1.2, 3.4), simplex2d(gen2, 1.2, 3.4), 1e-12);
    try std.testing.expectApproxEqAbs(simplex3d(gen1, 1.2, 3.4, 5.6), simplex3d(gen2, 1.2, 3.4, 5.6), 1e-12);
}

test "perlin2d stays roughly within [-1, 1]" {
    const g = Generator.init(12_345);
    var min: f64 = std.math.inf(f64);
    var max: f64 = -std.math.inf(f64);

    var x: f64 = 0.0;
    while (x < 10.0) : (x += 0.47) {
        var y: f64 = 0.0;
        while (y < 10.0) : (y += 0.53) {
            const v = perlin2d(g, x, y);
            min = @min(min, v);
            max = @max(max, v);
        }
    }

    try std.testing.expect(min >= -1.0 and max <= 1.0);
}

test "perlin3d stays roughly within [-1, 1]" {
    const g = Generator.init(12_345);
    var min: f64 = std.math.inf(f64);
    var max: f64 = -std.math.inf(f64);

    var x: f64 = 0.0;
    while (x < 5.0) : (x += 0.61) {
        var y: f64 = 0.0;
        while (y < 5.0) : (y += 0.67) {
            var z: f64 = 0.0;
            while (z < 5.0) : (z += 0.71) {
                const v = perlin3d(g, x, y, z);
                min = @min(min, v);
                max = @max(max, v);
            }
        }
    }

    try std.testing.expect(min >= -1.0 and max <= 1.0);
}

test "simplex2d stays roughly within [-1, 1]" {
    const g = Generator.init(12_345);
    var min: f64 = std.math.inf(f64);
    var max: f64 = -std.math.inf(f64);

    var x: f64 = 0.0;
    while (x < 10.0) : (x += 0.47) {
        var y: f64 = 0.0;
        while (y < 10.0) : (y += 0.53) {
            const v = simplex2d(g, x, y);
            min = @min(min, v);
            max = @max(max, v);
        }
    }

    try std.testing.expect(min >= -1.0 and max <= 1.0);
}

test "simplex3d stays roughly within [-1, 1]" {
    const g = Generator.init(12_345);
    var min: f64 = std.math.inf(f64);
    var max: f64 = -std.math.inf(f64);

    var x: f64 = 0.0;
    while (x < 5.0) : (x += 0.61) {
        var y: f64 = 0.0;
        while (y < 5.0) : (y += 0.67) {
            var z: f64 = 0.0;
            while (z < 5.0) : (z += 0.71) {
                const v = simplex3d(g, x, y, z);
                min = @min(min, v);
                max = @max(max, v);
            }
        }
    }

    try std.testing.expect(min >= -1.0 and max <= 1.0);
}
