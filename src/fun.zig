pub const gamma = @import("fun/gamma.zig");
pub const digamma = @import("fun/digamma.zig");
pub const erf = @import("fun/erf.zig");
pub const bessel = @import("fun/bessel.zig");
pub const mod_bessel = @import("fun/mod_bessel.zig");
pub const misc = @import("fun/misc.zig");
pub const interp = @import("fun/interp.zig");
pub const sinusoid = @import("fun/sinusoid.zig");
pub const extra = @import("fun/extra.zig");
pub const complex = @import("fun/complex.zig");
pub const cgamma = @import("fun/cgamma.zig");

test {
    _ = gamma;
    _ = digamma;
    _ = erf;
    _ = bessel;
    _ = mod_bessel;
    _ = misc;
    _ = interp;
    _ = sinusoid;
    _ = extra;
    _ = complex;
    _ = cgamma;
}
