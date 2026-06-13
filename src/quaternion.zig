const std = @import("std");
const math = std.math;

/// Tolerance used by `exp`, `log`, and `sqrt` to avoid division by near-zero.
const epsilon: f64 = 1e-14;

/// Quaternion with scalar part `w` and vector part `(x, y, z)`.
pub const Quaternion = struct {
    w: f64,
    x: f64,
    y: f64,
    z: f64,

    const Self = @This();

    /// Returns a copy of the quaternion.
    pub fn copy(self: Self) Self {
        return quaternion(self.w, self.x, self.y, self.z);
    }

    /// Formats the quaternion as `w + xi + yj + zk`.
    pub fn format(
        self: Self,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{d} + {d}i + {d}j + {d}k", .{ self.w, self.x, self.y, self.z });
    }

    /// Component-wise addition.
    pub fn add(self: Self, other: Self) Self {
        return quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z);
    }

    /// Component-wise subtraction.
    pub fn sub(self: Self, other: Self) Self {
        return quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z);
    }

    /// Hamilton product.
    pub fn mul(self: Self, other: Self) Self {
        return quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        );
    }

    /// Quaternion division: `self / other`.
    pub fn div(self: Self, other: Self) Self {
        const other_norm = other.norm();
        return quaternion(
            (self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z) / other_norm,
            (-self.w * other.x + self.x * other.w - self.y * other.z + self.z * other.y) / other_norm,
            (-self.w * other.y + self.x * other.z + self.y * other.w - self.z * other.x) / other_norm,
            (-self.w * other.z - self.x * other.y + self.y * other.x + self.z * other.w) / other_norm,
        );
    }

    /// Adds a scalar to the scalar component.
    pub fn scalar_add(self: Self, s: f64) Self {
        return quaternion(s + self.w, self.x, self.y, self.z);
    }

    /// Subtracts a scalar from the scalar component.
    pub fn scalar_sub(self: Self, s: f64) Self {
        return quaternion(s - self.w, self.x, self.y, self.z);
    }

    /// Multiplies every component by a scalar.
    pub fn scale(self: Self, s: f64) Self {
        return quaternion(s * self.w, s * self.x, s * self.y, s * self.z);
    }

    /// Divides every component by the quaternion norm (not by `s`).
    pub fn scalar_div(self: Self, s: f64) Self {
        const self_norm = self.norm();
        return quaternion(
            (s * self.w) / self_norm,
            (-s * self.x) / self_norm,
            (-s * self.y) / self_norm,
            (-s * self.z) / self_norm,
        );
    }

    /// Negates every component.
    pub fn opposite(self: Self) Self {
        return quaternion(-self.w, -self.x, -self.y, -self.z);
    }

    /// Returns the conjugate `(w, -x, -y, -z)`.
    pub fn conjugate(self: Self) Self {
        return quaternion(self.w, -self.x, -self.y, -self.z);
    }

    /// Returns the squared norm (also called the quaternion norm).
    pub fn norm(self: Self) f64 {
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
    }

    /// Returns the Euclidean magnitude.
    pub fn abs(self: Self) f64 {
        return math.sqrt(self.norm());
    }

    /// Returns twice the magnitude of the logarithm (rotation angle for unit quaternions).
    pub fn angle(self: Self) f64 {
        return 2.0 * log(self).abs();
    }

    /// Returns a unit quaternion pointing in the same direction.
    /// If the magnitude is zero, returns the input unchanged.
    pub fn normalized(self: Self) Self {
        const self_abs = self.abs();
        if (self_abs == 0.0) {
            return self;
        }
        return quaternion(self.w / self_abs, self.x / self_abs, self.y / self_abs, self.z / self_abs);
    }

    /// Returns the multiplicative inverse.
    pub fn inverse(self: Self) Self {
        const self_norm = self.norm();
        return quaternion(self.w / self_norm, -self.x / self_norm, -self.y / self_norm, -self.z / self_norm);
    }

    /// Returns the dot (inner) product with another quaternion.
    pub fn dot(self: Self, other: Self) f64 {
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;
    }

    /// Returns true if any component is NaN.
    pub fn is_nan(self: Self) bool {
        return math.isNan(self.w) or math.isNan(self.x) or math.isNan(self.y) or math.isNan(self.z);
    }

    /// Returns true if every component is zero. A NaN-containing quaternion is
    /// considered zero to match VSL's semantics.
    pub fn is_zero(self: Self) bool {
        if (self.is_nan()) return true;
        return self.w == 0.0 and self.x == 0.0 and self.y == 0.0 and self.z == 0.0;
    }

    /// Returns true if any component is infinite.
    pub fn is_inf(self: Self) bool {
        return math.isInf(self.w) or math.isInf(self.x) or math.isInf(self.y) or math.isInf(self.z);
    }

    /// Returns true if every component is finite.
    pub fn is_finite(self: Self) bool {
        return is_finite_scalar(self.w) and is_finite_scalar(self.x) and is_finite_scalar(self.y) and is_finite_scalar(self.z);
    }

    /// Returns true if the quaternion equals the identity quaternion.
    pub fn is_identity(self: Self) bool {
        return self.w == 1.0 and self.x == 0.0 and self.y == 0.0 and self.z == 0.0;
    }

    /// Returns true if the two quaternions are exactly equal and neither is NaN.
    pub fn equal(self: Self, other: Self) bool {
        return !self.is_nan() and !other.is_nan() and
            self.w == other.w and self.x == other.x and self.y == other.y and self.z == other.z;
    }

    /// Component-wise ordering: less-than.
    pub fn is_less(self: Self, other: Self) bool {
        if (self.is_nan() or other.is_nan()) return false;
        if (self.w != other.w) return self.w < other.w;
        if (self.x != other.x) return self.x < other.x;
        if (self.y != other.y) return self.y < other.y;
        return self.z < other.z;
    }

    /// Component-wise ordering: greater-than.
    pub fn is_greater(self: Self, other: Self) bool {
        if (self.is_nan() or other.is_nan()) return false;
        if (self.w != other.w) return self.w > other.w;
        if (self.x != other.x) return self.x > other.x;
        if (self.y != other.y) return self.y > other.y;
        return self.z > other.z;
    }

    /// Component-wise ordering: less-than-or-equal.
    pub fn is_less_equal(self: Self, other: Self) bool {
        if (self.is_nan() or other.is_nan()) return false;
        if (self.w != other.w) return self.w < other.w;
        if (self.x != other.x) return self.x < other.x;
        if (self.y != other.y) return self.y < other.y;
        return self.z <= other.z;
    }

    /// Component-wise ordering: greater-than-or-equal.
    pub fn is_greater_equal(self: Self, other: Self) bool {
        if (self.is_nan() or other.is_nan()) return false;
        if (self.w != other.w) return self.w > other.w;
        if (self.x != other.x) return self.x > other.x;
        if (self.y != other.y) return self.y > other.y;
        return self.z >= other.z;
    }

    /// Raises this quaternion to a real scalar power `s`.
    /// Computes `exp(s * log(self))`, with the convention `0^0 = 1`.
    pub fn scalar_pow(self: Self, s: f64) Self {
        if (s == 0.0) {
            return if (self.is_zero()) quaternion(1.0, 0.0, 0.0, 0.0) else quaternion(0.0, 0.0, 0.0, 0.0);
        }
        return exp(log(self).scale(s));
    }

    /// Raises this quaternion to the power of another quaternion.
    pub fn pow(self: Self, other: Self) Self {
        if (self.is_zero()) {
            return if (other.is_zero()) quaternion(1.0, 0.0, 0.0, 0.0) else quaternion(0.0, 0.0, 0.0, 0.0);
        }
        return exp(log(self).mul(other));
    }

    /// Returns the exponential of the quaternion.
    pub fn exp(self: Self) Self {
        const vnorm = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
        if (vnorm > epsilon) {
            const s = math.sin(vnorm) / vnorm;
            const e = math.exp(self.w);
            return quaternion(e * math.cos(vnorm), e * s * self.x, e * s * self.y, e * s * self.z);
        } else {
            return quaternion(math.exp(self.w), 0.0, 0.0, 0.0);
        }
    }

    /// Returns the principal logarithm of the quaternion.
    pub fn log(self: Self) Self {
        const b = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
        if (@abs(b) <= epsilon * @abs(self.w)) {
            if (self.w < 0.0) {
                if (@abs(self.w + 1.0) > epsilon) {
                    return quaternion(math.log(f64, math.e, -self.w), math.pi, 0.0, 0.0);
                } else {
                    return quaternion(0.0, math.pi, 0.0, 0.0);
                }
            } else {
                return quaternion(math.log(f64, math.e, self.w), 0.0, 0.0, 0.0);
            }
        } else {
            const v = math.atan2(b, self.w);
            const f = v / b;
            return quaternion(math.log(f64, math.e, self.w * self.w + b * b) / 2.0, f * self.x, f * self.y, f * self.z);
        }
    }

    /// Returns the principal square root of the quaternion.
    pub fn sqrt(self: Self) Self {
        const self_abs = self.abs();
        if (@abs(1.0 + self.w / self_abs) < epsilon * self_abs) {
            return quaternion(0.0, 1.0, 0.0, 0.0);
        } else {
            const c = math.sqrt(self_abs / (2.0 + 2.0 * self.w / self_abs));
            return quaternion(
                (1.0 + self.w / self_abs) * c,
                self.x * c / self_abs,
                self.y * c / self_abs,
                self.z * c / self_abs,
            );
        }
    }

    /// Linearly interpolates between this quaternion and another.
    pub fn lerp(self: Self, other: Self, tau: f64) Self {
        if (tau == 0.0) return self.copy();
        if (tau == 1.0) return other.copy();
        const f1 = 1.0 - tau;
        const f2 = tau;
        return quaternion(
            f1 * self.w + f2 * other.w,
            f1 * self.x + f2 * other.x,
            f1 * self.y + f2 * other.y,
            f1 * self.z + f2 * other.z,
        );
    }

    /// Normalized linear interpolation.
    pub fn nlerp(self: Self, other: Self, tau: f64) Self {
        return self.lerp(other, tau).normalized();
    }

    /// Spherical linear interpolation.
    pub fn slerp(self: Self, other: Self, tau: f64) Self {
        if (self.rotor_chordal_distance(other) <= math.sqrt2) {
            return other.div(self).scalar_pow(tau).mul(self);
        } else {
            return other.opposite().div(self).scalar_pow(tau).mul(self);
        }
    }

    /// Spherical quadrangle interpolation.
    pub fn squad(self: Self, taui: f64, ai: Self, bip1: Self, qip1: Self) Self {
        return self.slerp(qip1, taui).slerp(ai.slerp(bip1, taui), 2.0 * taui * (1.0 - taui));
    }

    /// Rotates a 3D vector by the quaternion using `q * v * q^-1`.
    /// For unit quaternions the inverse equals the conjugate.
    pub fn rotate_vector(self: Self, v: [3]f64) [3]f64 {
        const vq = quaternion(0.0, v[0], v[1], v[2]);
        const r = self.mul(vq).mul(self.conjugate());
        return .{ r.x, r.y, r.z };
    }

    /// Rotor intrinsic (geodesic) distance between two quaternions.
    pub fn rotor_intrinsic_distance(self: Self, other: Self) f64 {
        return 2.0 * self.div(other).log().abs();
    }

    /// Rotor chordal distance between two quaternions.
    pub fn rotor_chordal_distance(self: Self, other: Self) f64 {
        return self.sub(other).abs();
    }

    /// Rotation intrinsic distance, accounting for `q == -q` equivalence.
    pub fn rotation_intrinsic_distance(self: Self, other: Self) f64 {
        if (self.rotor_chordal_distance(other) <= math.sqrt2) {
            return 2.0 * self.div(other).log().abs();
        } else {
            return 2.0 * self.div(other.opposite()).log().abs();
        }
    }

    /// Rotation chordal distance, accounting for `q == -q` equivalence.
    pub fn rotation_chordal_distance(self: Self, other: Self) f64 {
        if (self.rotor_chordal_distance(other) <= math.sqrt2) {
            return self.sub(other).abs();
        } else {
            return self.add(other).abs();
        }
    }

    /// Identity parity conjugate (no change).
    pub fn parity_conjugate(self: Self) Self {
        return quaternion(self.w, self.x, self.y, self.z);
    }

    /// Symmetric part under identity parity.
    pub fn parity_symmetric_part(self: Self) Self {
        return quaternion(self.w, self.x, self.y, self.z);
    }

    /// Antisymmetric part under identity parity.
    pub fn parity_antisymmetric_part(_: Self) Self {
        return quaternion(0.0, 0.0, 0.0, 0.0);
    }

    /// Parity conjugate across the X axis.
    pub fn x_parity_conjugate(self: Self) Self {
        return quaternion(self.w, self.x, -self.y, -self.z);
    }

    /// Symmetric part across the X axis.
    pub fn x_parity_symmetric_part(self: Self) Self {
        return quaternion(self.w, self.x, 0.0, 0.0);
    }

    /// Antisymmetric part across the X axis.
    pub fn x_parity_antisymmetric_part(self: Self) Self {
        return quaternion(0.0, 0.0, self.y, self.z);
    }

    /// Parity conjugate across the Y axis.
    pub fn y_parity_conjugate(self: Self) Self {
        return quaternion(self.w, -self.x, self.y, -self.z);
    }

    /// Symmetric part across the Y axis.
    pub fn y_parity_symmetric_part(self: Self) Self {
        return quaternion(self.w, 0.0, self.y, 0.0);
    }

    /// Antisymmetric part across the Y axis.
    pub fn y_parity_antisymmetric_part(self: Self) Self {
        return quaternion(0.0, self.x, 0.0, self.z);
    }

    /// Parity conjugate across the Z axis.
    pub fn z_parity_conjugate(self: Self) Self {
        return quaternion(self.w, -self.x, -self.y, self.z);
    }

    /// Symmetric part across the Z axis.
    pub fn z_parity_symmetric_part(self: Self) Self {
        return quaternion(self.w, 0.0, 0.0, self.z);
    }

    /// Antisymmetric part across the Z axis.
    pub fn z_parity_antisymmetric_part(self: Self) Self {
        return quaternion(0.0, self.x, self.y, 0.0);
    }
};

/// Constructs a quaternion from its four components.
pub inline fn quaternion(w: f64, x: f64, y: f64, z: f64) Quaternion {
    return .{ .w = w, .x = x, .y = y, .z = z };
}

/// Returns the identity quaternion (no rotation).
pub inline fn id() Quaternion {
    return quaternion(1.0, 0.0, 0.0, 0.0);
}

/// Constructs a unit quaternion representing a rotation of `radians` radians
/// around the axis `(x, y, z)`.
pub fn from_axis_angle(radians: f64, x: f64, y: f64, z: f64) Quaternion {
    const half_angle = radians / 2.0;
    const s = math.sin(half_angle);
    const c = math.cos(half_angle);
    return quaternion(c, x * s, y * s, z * s).normalized();
}

/// Constructs a quaternion from spherical coordinates.
pub fn from_spherical_coords(theta: f64, phi: f64) Quaternion {
    const half_theta = theta / 2.0;
    const half_phi = phi / 2.0;
    const ct = math.cos(half_theta);
    const cp = math.cos(half_phi);
    const st = math.sin(half_theta);
    const sp = math.sin(half_phi);
    return quaternion(cp * ct, -sp * st, st * cp, sp * ct);
}

/// Constructs a quaternion from Euler angles (Tait-Bryan convention).
pub fn from_euler_angles(alpha: f64, beta: f64, gamma: f64) Quaternion {
    const half_alpha = alpha / 2.0;
    const half_beta = beta / 2.0;
    const half_gamma = gamma / 2.0;
    const ca = math.cos(half_alpha);
    const cb = math.cos(half_beta);
    const cc = math.cos(half_gamma);
    const sa = math.sin(half_alpha);
    const sb = math.sin(half_beta);
    const sc = math.sin(half_gamma);
    return quaternion(
        ca * cb * cc - sa * cb * sc,
        ca * sb * sc - sa * sb * cc,
        ca * sb * cc + sa * sb * sc,
        sa * cb * cc + ca * cb * sc,
    );
}

/// Converts a 3x3 rotation matrix into a quaternion.
pub fn from_matrix(m: [3][3]f64) Quaternion {
    const trace = m[0][0] + m[1][1] + m[2][2];
    if (trace > 0.0) {
        const s = 0.5 / math.sqrt(trace + 1.0);
        return quaternion(
            0.25 / s,
            (m[2][1] - m[1][2]) * s,
            (m[0][2] - m[2][0]) * s,
            (m[1][0] - m[0][1]) * s,
        );
    } else if (m[0][0] > m[1][1] and m[0][0] > m[2][2]) {
        const s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        return quaternion(
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        );
    } else if (m[1][1] > m[2][2]) {
        const s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]);
        return quaternion(
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        );
    } else {
        const s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]);
        return quaternion(
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        );
    }
}

/// Converts a quaternion into a 3x3 rotation matrix.
pub fn to_matrix(q: Quaternion) [3][3]f64 {
    const w = q.w;
    const x = q.x;
    const y = q.y;
    const z = q.z;
    return .{
        .{ 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y) },
        .{ 2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x) },
        .{ 2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y) },
    };
}

fn is_finite_scalar(a: f64) bool {
    return !math.isNan(a) and !math.isInf(a);
}

test "constructors produce expected components" {
    const i = id();
    try std.testing.expectEqual(1.0, i.w);
    try std.testing.expectEqual(0.0, i.x);
    try std.testing.expectEqual(0.0, i.y);
    try std.testing.expectEqual(0.0, i.z);

    const q = quaternion(1.0, 2.0, 3.0, 4.0);
    try std.testing.expectEqual(1.0, q.w);
    try std.testing.expectEqual(2.0, q.x);
    try std.testing.expectEqual(3.0, q.y);
    try std.testing.expectEqual(4.0, q.z);

    const axis_angle = from_axis_angle(math.pi / 2.0, 0.0, 0.0, 1.0);
    try std.testing.expectApproxEqAbs(1.0 / math.sqrt2, axis_angle.w, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, axis_angle.x, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, axis_angle.y, 1e-12);
    try std.testing.expectApproxEqAbs(1.0 / math.sqrt2, axis_angle.z, 1e-12);
}

test "basic algebra" {
    const a = quaternion(1.0, 2.0, 3.0, 4.0);
    const b = quaternion(5.0, 6.0, 7.0, 8.0);

    const s = a.add(b);
    try std.testing.expectEqual(6.0, s.w);
    try std.testing.expectEqual(8.0, s.x);
    try std.testing.expectEqual(10.0, s.y);
    try std.testing.expectEqual(12.0, s.z);

    const d = a.sub(b);
    try std.testing.expectEqual(-4.0, d.w);
    try std.testing.expectEqual(-4.0, d.x);
    try std.testing.expectEqual(-4.0, d.y);
    try std.testing.expectEqual(-4.0, d.z);

    const p = a.mul(b);
    try std.testing.expectEqual(-60.0, p.w);
    try std.testing.expectEqual(12.0, p.x);
    try std.testing.expectEqual(30.0, p.y);
    try std.testing.expectEqual(24.0, p.z);

    const sc = a.scale(2.0);
    try std.testing.expectEqual(2.0, sc.w);
    try std.testing.expectEqual(4.0, sc.x);
    try std.testing.expectEqual(6.0, sc.y);
    try std.testing.expectEqual(8.0, sc.z);

    const c = a.conjugate();
    try std.testing.expectEqual(1.0, c.w);
    try std.testing.expectEqual(-2.0, c.x);
    try std.testing.expectEqual(-3.0, c.y);
    try std.testing.expectEqual(-4.0, c.z);

    try std.testing.expectEqual(30.0, a.norm());
    try std.testing.expectApproxEqAbs(math.sqrt(30.0), a.abs(), 1e-12);

    const inv = a.inverse();
    const one = a.mul(inv);
    try std.testing.expectApproxEqAbs(1.0, one.w, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, one.x, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, one.y, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, one.z, 1e-12);

    try std.testing.expectEqual(70.0, a.dot(b));
}

test "classification" {
    const q = quaternion(1.0, 0.0, 0.0, 0.0);
    try std.testing.expect(q.is_identity());
    try std.testing.expect(q.is_finite());
    try std.testing.expect(!q.is_nan());

    const nan_q = quaternion(math.nan(f64), 0.0, 0.0, 0.0);
    try std.testing.expect(nan_q.is_nan());
    try std.testing.expect(nan_q.is_zero());
    try std.testing.expect(!nan_q.is_finite());

    const inf_q = quaternion(math.inf(f64), 0.0, 0.0, 0.0);
    try std.testing.expect(inf_q.is_inf());
    try std.testing.expect(!inf_q.is_finite());
}

test "rotation helpers" {
    // 90 degree rotation around Z maps (1,0,0) to (0,1,0).
    const q = from_axis_angle(math.pi / 2.0, 0.0, 0.0, 1.0);
    const v = q.rotate_vector(.{ 1.0, 0.0, 0.0 });
    try std.testing.expectApproxEqAbs(0.0, v[0], 1e-12);
    try std.testing.expectApproxEqAbs(1.0, v[1], 1e-12);
    try std.testing.expectApproxEqAbs(0.0, v[2], 1e-12);

    // nlerp between identity and the Z rotation at tau=0.5 should stay unit.
    const mid = id().nlerp(q, 0.5);
    try std.testing.expectApproxEqAbs(1.0, mid.abs(), 1e-12);

    // slerp between identity and itself returns identity.
    const same = id().slerp(id(), 0.5);
    try std.testing.expectApproxEqAbs(1.0, same.w, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, same.x, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, same.y, 1e-12);
    try std.testing.expectApproxEqAbs(0.0, same.z, 1e-12);
}

test "matrix conversion round-trips" {
    const q = from_axis_angle(math.pi / 3.0, 1.0, 1.0, 1.0);
    const m = to_matrix(q);
    const back = from_matrix(m);
    try std.testing.expectApproxEqAbs(@abs(q.w), @abs(back.w), 1e-12);
    try std.testing.expectApproxEqAbs(q.x, back.x, 1e-12);
    try std.testing.expectApproxEqAbs(q.y, back.y, 1e-12);
    try std.testing.expectApproxEqAbs(q.z, back.z, 1e-12);
}
