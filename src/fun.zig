pub const gamma = @import("fun/gamma.zig");
pub const digamma = @import("fun/digamma.zig");
pub const erf = @import("fun/erf.zig");
pub const bessel = @import("fun/bessel.zig");
pub const mod_bessel = @import("fun/mod_bessel.zig");
pub const misc = @import("fun/misc.zig");
pub const interp = @import("fun/interp.zig");
pub const extra = @import("fun/extra.zig");

test {
    _ = gamma;
    _ = digamma;
    _ = erf;
    _ = bessel;
    _ = mod_bessel;
    _ = misc;
    _ = interp;
    _ = extra;
}
