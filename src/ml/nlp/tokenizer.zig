const std = @import("std");
const Error = @import("../../errors.zig").Error;

/// Configuration for `tokenize`.
pub const TokenizeConfig = struct {
    lowercase: bool = true,
    remove_punctuation: bool = true,
    ngram_min: usize = 1,
    ngram_max: usize = 1,
};

/// Returns true for standard ASCII punctuation characters.
pub fn is_punctuation(c: u8) bool {
    return switch (c) {
        '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~' => true,
        else => false,
    };
}

/// Tokenizes `text` according to `config`.
///
/// The returned slice and every token string are allocated with `allocator` and
/// must be freed by the caller (free each token, then the outer slice).
pub fn tokenize(allocator: std.mem.Allocator, text: []const u8, config: TokenizeConfig) Error![][]const u8 {
    if (text.len == 0) return &[_][]const u8{};

    var normalized = try allocator.alloc(u8, text.len);
    defer allocator.free(normalized);

    var i: usize = 0;
    for (text) |c| {
        var ch = c;
        if (config.lowercase) ch = std.ascii.toLower(ch);
        if (config.remove_punctuation and is_punctuation(ch)) {
            normalized[i] = ' ';
        } else {
            normalized[i] = ch;
        }
        i += 1;
    }

    var token_list = std.ArrayList([]const u8).empty;
    defer {
        for (token_list.items) |tok| allocator.free(tok);
        token_list.deinit(allocator);
    }

    var it = std.mem.splitScalar(u8, normalized, ' ');
    while (it.next()) |raw| {
        if (raw.len == 0) continue;
        const tok = try allocator.dupe(u8, raw);
        errdefer allocator.free(tok);
        try token_list.append(allocator, tok);
    }

    if (token_list.items.len == 0) return &[_][]const u8{};

    const min_n = @min(config.ngram_min, config.ngram_max);
    const max_n = @max(config.ngram_min, config.ngram_max);

    if (min_n == 1 and max_n == 1) {
        return try token_list.toOwnedSlice(allocator);
    }

    var ngram_list = std.ArrayList([]const u8).empty;
    defer {
        for (ngram_list.items) |tok| allocator.free(tok);
        ngram_list.deinit(allocator);
    }

    const base = token_list.items;
    for (min_n..max_n + 1) |n| {
        if (n > base.len) continue;
        for (0..base.len - n + 1) |start| {
            if (n == 1) {
                const tok = try allocator.dupe(u8, base[start]);
                errdefer allocator.free(tok);
                try ngram_list.append(allocator, tok);
            } else {
                var joined = std.ArrayList(u8).empty;
                defer joined.deinit(allocator);
                for (base[start .. start + n]) |part| {
                    if (joined.items.len > 0) try joined.append(allocator, '_');
                    try joined.appendSlice(allocator, part);
                }
                const tok = try joined.toOwnedSlice(allocator);
                errdefer allocator.free(tok);
                try ngram_list.append(allocator, tok);
            }
        }
    }

    return try ngram_list.toOwnedSlice(allocator);
}

test "tokenize basic lowercase and punctuation removal" {
    const allocator = std.testing.allocator;
    const text = "Hello, world!";
    const config = TokenizeConfig{ .lowercase = true, .remove_punctuation = true };
    const tokens = try tokenize(allocator, text, config);
    defer {
        for (tokens) |tok| allocator.free(tok);
        allocator.free(tokens);
    }
    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqualStrings("hello", tokens[0]);
    try std.testing.expectEqualStrings("world", tokens[1]);
}

test "tokenize preserves case and punctuation" {
    const allocator = std.testing.allocator;
    const text = "Hello, world!";
    const config = TokenizeConfig{ .lowercase = false, .remove_punctuation = false };
    const tokens = try tokenize(allocator, text, config);
    defer {
        for (tokens) |tok| allocator.free(tok);
        allocator.free(tokens);
    }
    try std.testing.expectEqual(@as(usize, 2), tokens.len);
    try std.testing.expectEqualStrings("Hello,", tokens[0]);
    try std.testing.expectEqualStrings("world!", tokens[1]);
}

test "tokenize ngrams" {
    const allocator = std.testing.allocator;
    const text = "a b c";
    const config = TokenizeConfig{ .ngram_min = 1, .ngram_max = 2 };
    const tokens = try tokenize(allocator, text, config);
    defer {
        for (tokens) |tok| allocator.free(tok);
        allocator.free(tokens);
    }
    try std.testing.expectEqual(@as(usize, 5), tokens.len);
    try std.testing.expectEqualStrings("a", tokens[0]);
    try std.testing.expectEqualStrings("b", tokens[1]);
    try std.testing.expectEqualStrings("c", tokens[2]);
    try std.testing.expectEqualStrings("a_b", tokens[3]);
    try std.testing.expectEqualStrings("b_c", tokens[4]);
}

test "tokenize empty input" {
    const allocator = std.testing.allocator;
    const tokens = try tokenize(allocator, "", TokenizeConfig{});
    defer {
        for (tokens) |tok| allocator.free(tok);
        allocator.free(tokens);
    }
    try std.testing.expectEqual(@as(usize, 0), tokens.len);
}

test "is_punctuation" {
    try std.testing.expect(is_punctuation('!'));
    try std.testing.expect(is_punctuation('.'));
    try std.testing.expect(is_punctuation('\''));
    try std.testing.expect(!is_punctuation('a'));
    try std.testing.expect(!is_punctuation(' '));
}
