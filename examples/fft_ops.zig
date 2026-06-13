const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const sample_rate: f64 = 64.0;
    const n: usize = 64;

    // Build a real signal as a sum of two sinusoids.
    var signal: [n]f64 = undefined;
    for (0..n) |i| {
        const t = @as(f64, @floatFromInt(i)) / sample_rate;
        signal[i] = @sin(2.0 * std.math.pi * 5.0 * t) +
            0.5 * @sin(2.0 * std.math.pi * 12.0 * t);
    }

    std.debug.print("Original signal:\n", .{});
    for (signal) |v| {
        std.debug.print("{d:.6} ", .{v});
    }
    std.debug.print("\n\n", .{});

    // Forward FFT.
    const spectrum = try zsl.fft.fft(&signal, allocator);
    defer allocator.free(spectrum);

    // Magnitude spectrum.
    const mag = try zsl.fft.magnitude_spectrum(spectrum, allocator);
    defer allocator.free(mag);

    std.debug.print("Magnitude spectrum (first {d} bins):\n", .{n / 2 + 1});
    for (mag, 0..) |v, k| {
        if (k > n / 2) break;
        std.debug.print("bin {d}: {d:.6}\n", .{ k, v });
    }

    // Inverse FFT normalized to recover the original signal.
    const reconstructed = try zsl.fft.ifft_normalized(spectrum, allocator);
    defer allocator.free(reconstructed);

    std.debug.print("\nReconstructed signal:\n", .{});
    for (reconstructed) |v| {
        std.debug.print("{d:.6} ", .{v});
    }
    std.debug.print("\n", .{});

    var max_error: f64 = 0.0;
    for (reconstructed, signal) |r, s| {
        const err = @abs(r - s);
        if (err > max_error) max_error = err;
    }
    std.debug.print("\nMax reconstruction error: {e:.3}\n", .{max_error});

    for (reconstructed, signal) |r, s| {
        if (!zsl.float.approxEqAbs(f64, r, s, 1e-12)) {
            return error.NotConverged;
        }
    }
    std.debug.print("Reconstruction verified.\n", .{});
}
