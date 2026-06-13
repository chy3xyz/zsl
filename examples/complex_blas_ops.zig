const std = @import("std");
const zsl = @import("zsl");
const Complex = zsl.fun.complex.Complex;

const C64 = Complex(f64);
const caxpy = zsl.blas.complex.caxpy;
const cdotu = zsl.blas.complex.cdotu;
const cdotc = zsl.blas.complex.cdotc;
const cgemv = zsl.blas.complex.cgemv;
const cgemm = zsl.blas.complex.cgemm;

fn approxEqC64(actual: C64, expected: C64, tol: f64) bool {
    return @abs(actual.re - expected.re) <= tol and @abs(actual.im - expected.im) <= tol;
}

fn printComplex(label: []const u8, z: C64) void {
    std.debug.print("{s} = {d:.2} + {d:.2}i\n", .{ label, z.re, z.im });
}

pub fn main() !void {
    std.debug.print("=== Complex BLAS scaffolding demo ===\n\n", .{});

    // ------------------------------------------------------------------------
    // caxpy: y <- alpha*x + y
    // ------------------------------------------------------------------------
    var y = [_]C64{
        C64.new(1, 1),
        C64.new(2, -1),
        C64.new(0, 0),
    };
    const x = [_]C64{
        C64.new(1, 0),
        C64.new(0, 1),
        C64.new(2, 2),
    };
    const alpha = C64.new(2, 1);
    try caxpy(alpha, &x, &y);
    std.debug.print("caxpy result:\n", .{});
    for (0..y.len) |i| printComplex("  y[i]", y[i]);
    std.debug.print("\n", .{});

    // ------------------------------------------------------------------------
    // cdotu / cdotc
    // ------------------------------------------------------------------------
    const u = [_]C64{ C64.new(1, 2), C64.new(3, -1) };
    const v = [_]C64{ C64.new(2, 0), C64.new(1, 1) };
    const dotu = try cdotu(&u, &v);
    const dotc = try cdotc(&u, &v);
    printComplex("cdotu(x, y)", dotu);
    printComplex("cdotc(x, y)", dotc);
    std.debug.print("\n", .{});

    // ------------------------------------------------------------------------
    // cgemv: y <- A*x
    // ------------------------------------------------------------------------
    var A = [_]C64{
        C64.new(1, 1),  C64.new(2, 0),
        C64.new(0, -1), C64.new(3, 2),
    };
    const xv = [_]C64{ C64.new(1, 1), C64.new(2, -1) };
    var yv = [_]C64{ C64.new(0, 0), C64.new(0, 0) };
    try cgemv(C64.new(1, 0), &A, 2, &xv, C64.new(0, 0), &yv, 2, 2, false);
    std.debug.print("cgemv result:\n", .{});
    printComplex("  y[0]", yv[0]);
    printComplex("  y[1]", yv[1]);
    std.debug.print("\n", .{});

    // ------------------------------------------------------------------------
    // cgemm: C <- A*B
    // ------------------------------------------------------------------------
    var Am = [_]C64{
        C64.new(1, 1), C64.new(2, 0),
        C64.new(3, 0), C64.new(1, -1),
    };
    var Bm = [_]C64{
        C64.new(1, 0), C64.new(0, 1),
        C64.new(2, 1), C64.new(0, 0),
    };
    var Cm = [_]C64{
        C64.new(0, 0), C64.new(0, 0),
        C64.new(0, 0), C64.new(0, 0),
    };
    try cgemm(C64.new(1, 0), &Am, 2, &Bm, 2, C64.new(0, 0), &Cm, 2, 2, 2, 2);
    std.debug.print("cgemm result:\n", .{});
    printComplex("  C[0,0]", Cm[0]);
    printComplex("  C[0,1]", Cm[1]);
    printComplex("  C[1,0]", Cm[2]);
    printComplex("  C[1,1]", Cm[3]);

    // ------------------------------------------------------------------------
    // Sanity checks against hand-computed values
    // ------------------------------------------------------------------------
    std.debug.print("\n--- sanity checks ---\n", .{});
    std.debug.assert(approxEqC64(y[0], C64.new(3, 2), 1e-9));
    std.debug.assert(approxEqC64(y[1], C64.new(1, 1), 1e-9));
    std.debug.assert(approxEqC64(y[2], C64.new(2, 6), 1e-9));
    std.debug.assert(approxEqC64(dotu, C64.new(6, 6), 1e-9));
    std.debug.assert(approxEqC64(dotc, C64.new(4, 0), 1e-9));
    std.debug.assert(approxEqC64(yv[0], C64.new(4, 0), 1e-9));
    std.debug.assert(approxEqC64(yv[1], C64.new(9, 0), 1e-9));
    std.debug.assert(approxEqC64(Cm[0], C64.new(5, 3), 1e-9));
    std.debug.assert(approxEqC64(Cm[1], C64.new(-1, 1), 1e-9));
    std.debug.assert(approxEqC64(Cm[2], C64.new(6, -1), 1e-9));
    std.debug.assert(approxEqC64(Cm[3], C64.new(0, 3), 1e-9));
    std.debug.print("All sanity checks passed.\n", .{});
}
