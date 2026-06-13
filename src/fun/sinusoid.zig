const std = @import("std");

/// Sinusoid function.
///
///   y(t) = A * sin(2 * pi * f * t + phase) + offset
///
/// where A is the amplitude, f is the frequency, phase is the phase shift in
/// radians, and offset is the vertical offset.
pub const Sinusoid = struct {
    amplitude: f64,
    frequency: f64,
    phase: f64,
    offset: f64,

    /// Create a new sinusoid.
    pub fn init(amplitude: f64, frequency: f64, phase: f64, offset: f64) Sinusoid {
        return .{
            .amplitude = amplitude,
            .frequency = frequency,
            .phase = phase,
            .offset = offset,
        };
    }

    /// Evaluate the sinusoid at time t.
    pub fn evaluate(self: Sinusoid, t: f64) f64 {
        return self.amplitude * @sin(2.0 * std.math.pi * self.frequency * t + self.phase) + self.offset;
    }
};

test "Sinusoid evaluate" {
    const float = @import("../float.zig");
    const s = Sinusoid.init(2.0, 1.0, 0.0, 1.0);

    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(0.0), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(0.25), 3.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(0.5), 1.0, 1e-12));
    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(0.75), -1.0, 1e-12));
}

test "Sinusoid with phase and offset" {
    const float = @import("../float.zig");
    const s = Sinusoid.init(1.0, 0.5, std.math.pi / 2.0, -1.0);

    // sin(2*pi*0.5*t + pi/2) = cos(pi*t); at t=0 this is 1, so y = 0.
    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(0.0), 0.0, 1e-12));
    // cos(pi) = -1, so y = -2.
    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(1.0), -2.0, 1e-12));
    // cos(2*pi) = 1, so y = 0.
    try std.testing.expect(float.approxEqAbs(f64, s.evaluate(2.0), 0.0, 1e-12));
}
