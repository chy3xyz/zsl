const std = @import("std");
const Error = @import("../../errors.zig").Error;

const Rule = struct {
    suffix: []const u8,
    replacement: []const u8,
    min_len: usize,
};

// A minimal, ordered set of Lancaster-style stemming rules.
const rules = &[_]Rule{
    .{ .suffix = "ational", .replacement = "ate", .min_len = 8 },
    .{ .suffix = "tional", .replacement = "tion", .min_len = 7 },
    .{ .suffix = "izer", .replacement = "ize", .min_len = 5 },
    .{ .suffix = "ator", .replacement = "ate", .min_len = 5 },
    .{ .suffix = "ies", .replacement = "y", .min_len = 4 },
    .{ .suffix = "ied", .replacement = "y", .min_len = 4 },
    .{ .suffix = "ing", .replacement = "", .min_len = 4 },
    .{ .suffix = "ed", .replacement = "", .min_len = 3 },
    .{ .suffix = "s", .replacement = "", .min_len = 3 },
};

/// Stems `word` using a small set of Lancaster-style suffix rules.
///
/// The returned string is allocated with `allocator` and must be freed by the caller.
pub fn stem(allocator: std.mem.Allocator, word: []const u8) Error![]const u8 {
    for (rules) |rule| {
        if (word.len >= rule.min_len and std.mem.endsWith(u8, word, rule.suffix)) {
            const base_len = word.len - rule.suffix.len;
            const result_len = base_len + rule.replacement.len;
            const result = try allocator.alloc(u8, result_len);
            @memcpy(result[0..base_len], word[0..base_len]);
            @memcpy(result[base_len..], rule.replacement);
            return result;
        }
    }
    return try allocator.dupe(u8, word);
}

test "stem removes simple suffixes" {
    const allocator = std.testing.allocator;

    const walked = try stem(allocator, "walked");
    defer allocator.free(walked);
    try std.testing.expectEqualStrings("walk", walked);

    const running = try stem(allocator, "running");
    defer allocator.free(running);
    try std.testing.expectEqualStrings("runn", running);

    const cats = try stem(allocator, "cats");
    defer allocator.free(cats);
    try std.testing.expectEqualStrings("cat", cats);
}

test "stem ies and ied rules" {
    const allocator = std.testing.allocator;

    const flies = try stem(allocator, "flies");
    defer allocator.free(flies);
    try std.testing.expectEqualStrings("fly", flies);

    const tried = try stem(allocator, "tried");
    defer allocator.free(tried);
    try std.testing.expectEqualStrings("try", tried);
}

test "stem longer suffixes" {
    const allocator = std.testing.allocator;

    const national = try stem(allocator, "national");
    defer allocator.free(national);
    try std.testing.expectEqualStrings("nate", national);

    const rational = try stem(allocator, "rational");
    defer allocator.free(rational);
    try std.testing.expectEqualStrings("rate", rational);

    const optimizer = try stem(allocator, "optimizer");
    defer allocator.free(optimizer);
    try std.testing.expectEqualStrings("optimize", optimizer);
}

test "stem leaves short words unchanged" {
    const allocator = std.testing.allocator;

    const is = try stem(allocator, "is");
    defer allocator.free(is);
    try std.testing.expectEqualStrings("is", is);
}
