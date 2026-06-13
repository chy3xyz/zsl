const std = @import("std");
const la = @import("../../la.zig");
const Error = @import("../../errors.zig").Error;
const tokenizer = @import("tokenizer.zig");
const TokenizeConfig = tokenizer.TokenizeConfig;

const VocabEntry = struct {
    word: []const u8,
    count: usize,
};

/// `TfIdfVectorizer` converts a corpus into a document-term TF-IDF weight matrix.
pub const TfIdfVectorizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(usize),
    idf: []f64,
    max_features: usize,
    tokenize_config: TokenizeConfig,

    const Self = @This();

    /// Creates a new TF-IDF vectorizer. `max_features` of 0 means unlimited vocabulary.
    pub fn init(allocator: std.mem.Allocator, max_features: usize) Self {
        return .{
            .allocator = allocator,
            .vocab = std.StringHashMap(usize).init(allocator),
            .idf = &[_]f64{},
            .max_features = max_features,
            .tokenize_config = TokenizeConfig{},
        };
    }

    /// Releases all memory owned by the vectorizer.
    pub fn deinit(self: *Self) void {
        var it = self.vocab.keyIterator();
        while (it.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
        self.vocab.deinit();
        self.allocator.free(self.idf);
        self.idf = &[_]f64{};
    }

    /// Sets the tokenization configuration used during fitting and transformation.
    pub fn setTokenizeConfig(self: *Self, config: TokenizeConfig) void {
        self.tokenize_config = config;
    }

    /// Fits the vocabulary and IDF weights on `corpus` and returns the TF-IDF matrix.
    pub fn fit_transform(self: *Self, allocator: std.mem.Allocator, corpus: []const []const u8) Error!la.Matrix(f64) {
        // Clear previous state.
        var vit = self.vocab.keyIterator();
        while (vit.next()) |key_ptr| {
            self.allocator.free(key_ptr.*);
        }
        self.vocab.clearRetainingCapacity();
        self.allocator.free(self.idf);
        self.idf = &[_]f64{};

        const n_docs = corpus.len;
        if (n_docs == 0) return error.InvalidDimension;

        var counts = std.StringHashMap(usize).init(self.allocator);
        defer {
            var cit = counts.keyIterator();
            while (cit.next()) |key_ptr| {
                self.allocator.free(key_ptr.*);
            }
            counts.deinit();
        }

        var doc_freq = std.StringHashMap(usize).init(self.allocator);
        defer {
            var dit = doc_freq.keyIterator();
            while (dit.next()) |key_ptr| {
                self.allocator.free(key_ptr.*);
            }
            doc_freq.deinit();
        }

        for (corpus) |doc| {
            const tokens = try tokenizer.tokenize(self.allocator, doc, self.tokenize_config);
            defer {
                for (tokens) |tok| self.allocator.free(tok);
                self.allocator.free(tokens);
            }

            var seen = std.StringHashMap(void).init(self.allocator);
            defer {
                var sit = seen.keyIterator();
                while (sit.next()) |key_ptr| {
                    self.allocator.free(key_ptr.*);
                }
                seen.deinit();
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

                const seen_gop = try seen.getOrPut(tok);
                if (!seen_gop.found_existing) {
                    const owned = try self.allocator.dupe(u8, tok);
                    errdefer self.allocator.free(owned);
                    seen_gop.key_ptr.* = owned;

                    const df_gop = try doc_freq.getOrPut(tok);
                    if (df_gop.found_existing) {
                        df_gop.value_ptr.* += 1;
                    } else {
                        const df_owned = try self.allocator.dupe(u8, tok);
                        errdefer self.allocator.free(df_owned);
                        df_gop.key_ptr.* = df_owned;
                        df_gop.value_ptr.* = 1;
                    }
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
        self.idf = try self.allocator.alloc(f64, limit);
        errdefer {
            self.allocator.free(self.idf);
            self.idf = &[_]f64{};
        }

        const n_f: f64 = @floatFromInt(n_docs);
        for (0..limit) |i| {
            const word = entries[i].word;
            const owned = try self.allocator.dupe(u8, word);
            errdefer self.allocator.free(owned);
            try self.vocab.put(owned, i);

            const df = doc_freq.get(word) orelse 0;
            const df_f: f64 = @floatFromInt(df);
            self.idf[i] = @log(n_f / (1.0 + df_f));
        }

        var matrix = try la.Matrix(f64).init(allocator, n_docs, limit);
        errdefer matrix.deinit(allocator);

        for (corpus, 0..) |doc, row_idx| {
            const tokens = try tokenizer.tokenize(allocator, doc, self.tokenize_config);
            defer {
                for (tokens) |tok| allocator.free(tok);
                allocator.free(tokens);
            }

            var local_counts = std.StringHashMap(usize).init(allocator);
            defer {
                var lit = local_counts.keyIterator();
                while (lit.next()) |key_ptr| {
                    allocator.free(key_ptr.*);
                }
                local_counts.deinit();
            }

            for (tokens) |tok| {
                const gop = try local_counts.getOrPut(tok);
                if (gop.found_existing) {
                    gop.value_ptr.* += 1;
                } else {
                    const owned = try allocator.dupe(u8, tok);
                    errdefer allocator.free(owned);
                    gop.key_ptr.* = owned;
                    gop.value_ptr.* = 1;
                }
            }

            var lit = local_counts.iterator();
            while (lit.next()) |entry| {
                if (self.vocab.get(entry.key_ptr.*)) |col_idx| {
                    const cnt = entry.value_ptr.*;
                    const tf = @log(1.0 + @as(f64, @floatFromInt(cnt)));
                    try matrix.set(row_idx, col_idx, tf * self.idf[col_idx]);
                }
            }
        }

        return matrix;
    }
};

fn compareByCountDesc(_: void, a: VocabEntry, b: VocabEntry) bool {
    if (a.count != b.count) return a.count > b.count;
    return std.mem.lessThan(u8, a.word, b.word);
}

test "TfIdfVectorizer basic" {
    const allocator = std.testing.allocator;
    const corpus = &[_][]const u8{
        "hello world",
        "hello hello world",
        "foo bar",
    };
    var tfidf = TfIdfVectorizer.init(allocator, 0);
    defer tfidf.deinit();

    var matrix = try tfidf.fit_transform(allocator, corpus);
    defer matrix.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), matrix.rows);
    try std.testing.expectEqual(@as(usize, 4), matrix.cols);

    // Common terms (hello, world) have zero IDF because they appear in all docs.
    // Rare terms (foo, bar) have positive IDF.
    const hello_tfidf = try matrix.get(0, tfidf.vocab.get("hello").?);
    const foo_tfidf = try matrix.get(2, tfidf.vocab.get("foo").?);
    try std.testing.expectEqual(@as(f64, 0.0), hello_tfidf);
    try std.testing.expect(foo_tfidf > 0.0);
}

test "TfIdfVectorizer max_features" {
    const allocator = std.testing.allocator;
    const corpus = &[_][]const u8{
        "a b c",
        "a b d",
        "a e f",
    };
    var tfidf = TfIdfVectorizer.init(allocator, 2);
    defer tfidf.deinit();

    var matrix = try tfidf.fit_transform(allocator, corpus);
    defer matrix.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), matrix.rows);
    try std.testing.expectEqual(@as(usize, 2), matrix.cols);
}

/// Convenience wrapper around `TfIdfVectorizer.fit_transform`.
pub fn tf_idf(allocator: std.mem.Allocator, corpus: []const []const u8) Error!la.Matrix(f64) {
    var vectorizer = TfIdfVectorizer.init(allocator, 0);
    defer vectorizer.deinit();
    return try vectorizer.fit_transform(allocator, corpus);
}
