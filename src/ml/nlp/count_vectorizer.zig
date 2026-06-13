const std = @import("std");
const la = @import("../../la.zig");
const Error = @import("../../errors.zig").Error;
const tokenizer = @import("tokenizer.zig");
const TokenizeConfig = tokenizer.TokenizeConfig;

const VocabEntry = struct {
    word: []const u8,
    count: usize,
};

/// `CountVectorizer` converts a corpus of text documents into a matrix of token counts.
pub const CountVectorizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(usize),
    max_features: usize,
    tokenize_config: TokenizeConfig,

    const Self = @This();

    /// Creates a new vectorizer. `max_features` of 0 means unlimited vocabulary.
    pub fn init(allocator: std.mem.Allocator, max_features: usize) Self {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(usize).init(allocator),
            .max_features = max_features,
            .tokenize_config = TokenizeConfig{},
        };
    }

    /// Releases all vocabulary memory owned by the vectorizer.
    pub fn deinit(self: *Self) void {
        var it = self.vocab.keyIterator();
        while (it.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
        self.vocab.deinit();
    }

    /// Sets the tokenization configuration used during fitting and transformation.
    pub fn setTokenizeConfig(self: *Self, config: TokenizeConfig) void {
        self.tokenize_config = config;
    }

    /// Builds the vocabulary from `corpus`.
    pub fn fit(self: *Self, corpus: []const []const u8) Error!void {
        // Clear any previous vocabulary.
        var it = self.vocab.keyIterator();
        while (it.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
        self.vocab.clearRetainingCapacity();

        var counts = std.StringHashMap(usize).init(self.allocator);
        defer {
            var cit = counts.keyIterator();
            while (cit.next()) |key_ptr| {
                self.allocator.free(key_ptr.*);
            }
            counts.deinit();
        }

        for (corpus) |doc| {
            const tokens = try tokenizer.tokenize(self.allocator, doc, self.tokenize_config);
            defer {
                for (tokens) |tok| self.allocator.free(tok);
                self.allocator.free(tokens);
            }
            for (tokens) |tok| {
                const gop = try counts.getOrPut(tok);
                if (gop.found_existing) {
                    gop.value_ptr.* += 1;
                } else {
                    const owned = try self.allocator.dupe(u8, tok);
                    errdefer self.allocator.free(owned);
                    gop.key_ptr.* = owned;
                    gop.value_ptr.* = 1;
                }
            }
        }

        const n_terms = counts.count();
        var entries = try self.allocator.alloc(VocabEntry, n_terms);
        defer self.allocator.free(entries);

        var idx: usize = 0;
        var cit = counts.iterator();
        while (cit.next()) |entry| {
            entries[idx] = .{ .word = entry.key_ptr.*, .count = entry.value_ptr.* };
            idx += 1;
        }

        std.mem.sort(VocabEntry, entries, {}, compareByCountDesc);

        const limit = if (self.max_features == 0) entries.len else @min(self.max_features, entries.len);
        for (0..limit) |i| {
            const owned = try self.allocator.dupe(u8, entries[i].word);
            errdefer self.allocator.free(owned);
            try self.vocab.put(owned, i);
        }
    }

    /// Transforms `corpus` into a document-term count matrix using the fitted vocabulary.
    pub fn transform(self: *Self, allocator: std.mem.Allocator, corpus: []const []const u8) Error!la.Matrix(f64) {
        if (self.vocab.count() == 0) return error.NotFitted;
        var matrix = try la.Matrix(f64).init(allocator, corpus.len, self.vocab.count());
        errdefer matrix.deinit(allocator);

        for (corpus, 0..) |doc, row_idx| {
            const tokens = try tokenizer.tokenize(allocator, doc, self.tokenize_config);
            defer {
                for (tokens) |tok| allocator.free(tok);
                allocator.free(tokens);
            }
            for (tokens) |tok| {
                if (self.vocab.get(tok)) |col_idx| {
                    const current = try matrix.get(row_idx, col_idx);
                    try matrix.set(row_idx, col_idx, current + 1.0);
                }
            }
        }
        return matrix;
    }

    /// Fits the vectorizer on `corpus` and returns the document-term count matrix.
    pub fn fit_transform(self: *Self, allocator: std.mem.Allocator, corpus: []const []const u8) Error!la.Matrix(f64) {
        try self.fit(corpus);
        return try self.transform(allocator, corpus);
    }
};

fn compareByCountDesc(_: void, a: VocabEntry, b: VocabEntry) bool {
    if (a.count != b.count) return a.count > b.count;
    return std.mem.lessThan(u8, a.word, b.word);
}

test "CountVectorizer basic" {
    const allocator = std.testing.allocator;
    const corpus = &[_][]const u8{
        "hello world",
        "hello hello world",
    };
    var cv = CountVectorizer.init(allocator, 0);
    defer cv.deinit();

    var matrix = try cv.fit_transform(allocator, corpus);
    defer matrix.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), matrix.rows);
    try std.testing.expectEqual(@as(usize, 2), matrix.cols);
    try std.testing.expectEqual(@as(f64, 1.0), try matrix.get(0, 0));
    try std.testing.expectEqual(@as(f64, 1.0), try matrix.get(0, 1));
    try std.testing.expectEqual(@as(f64, 2.0), try matrix.get(1, 0));
    try std.testing.expectEqual(@as(f64, 1.0), try matrix.get(1, 1));
}

test "CountVectorizer max_features" {
    const allocator = std.testing.allocator;
    const corpus = &[_][]const u8{
        "a b c",
        "a b d",
    };
    var cv = CountVectorizer.init(allocator, 2);
    defer cv.deinit();

    var matrix = try cv.fit_transform(allocator, corpus);
    defer matrix.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), matrix.rows);
    try std.testing.expectEqual(@as(usize, 2), matrix.cols);
}

test "CountVectorizer not fitted error" {
    const allocator = std.testing.allocator;
    const corpus = &[_][]const u8{"hello"};
    var cv = CountVectorizer.init(allocator, 0);
    defer cv.deinit();

    try std.testing.expectError(error.NotFitted, cv.transform(allocator, corpus));
}

/// Convenience wrapper around `CountVectorizer.fit_transform`.
/// `ngrams` sets both the minimum and maximum n-gram width; `most_frequent`
/// limits the vocabulary size (0 means unlimited).
pub fn count_vectorize(
    allocator: std.mem.Allocator,
    corpus: []const []const u8,
    ngrams: usize,
    most_frequent: usize,
) Error!la.Matrix(f64) {
    var vectorizer = CountVectorizer.init(allocator, most_frequent);
    defer vectorizer.deinit();
    vectorizer.setTokenizeConfig(.{
        .lowercase = true,
        .remove_punctuation = true,
        .ngram_min = ngrams,
        .ngram_max = ngrams,
    });
    return try vectorizer.fit_transform(allocator, corpus);
}

/// Returns the most frequent n-gram strings in `corpus`.
/// The returned slice and each string are allocated with `allocator` and must
/// be freed by the caller.
pub fn most_frequent_ngrams(
    allocator: std.mem.Allocator,
    corpus: []const []const u8,
    ngrams: usize,
    most_frequent: usize,
) Error![][]const u8 {
    var vectorizer = CountVectorizer.init(allocator, most_frequent);
    defer vectorizer.deinit();
    vectorizer.setTokenizeConfig(.{
        .lowercase = true,
        .remove_punctuation = true,
        .ngram_min = ngrams,
        .ngram_max = ngrams,
    });
    try vectorizer.fit(corpus);

    const vocab_size = vectorizer.vocab.count();
    var result = try allocator.alloc([]const u8, vocab_size);
    errdefer {
        for (result) |s| allocator.free(s);
        allocator.free(result);
    }

    var it = vectorizer.vocab.iterator();
    while (it.next()) |entry| {
        const idx = entry.value_ptr.*;
        result[idx] = try allocator.dupe(u8, entry.key_ptr.*);
    }
    return result;
}
