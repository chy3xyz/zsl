const std = @import("std");
const zsl = @import("zsl");

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const text = "Hello, world! This is a simple NLP example. Hello again, world.";
    std.debug.print("Original text:\n{s}\n\n", .{text});

    // Tokenization
    const config = zsl.ml.nlp.TokenizeConfig{
        .lowercase = true,
        .remove_punctuation = true,
        .ngram_min = 1,
        .ngram_max = 2,
    };
    const tokens = try zsl.ml.nlp.tokenize(allocator, text, config);
    defer {
        for (tokens) |tok| allocator.free(tok);
        allocator.free(tokens);
    }
    std.debug.print("Tokens:\n", .{});
    for (tokens) |tok| {
        std.debug.print("  {s}\n", .{tok});
    }
    std.debug.print("\n", .{});

    // Count vectorizer
    const corpus = &[_][]const u8{
        "hello world",
        "hello nlp example",
        "nlp is fun",
    };
    var cv = zsl.ml.nlp.CountVectorizer.init(allocator, 0);
    defer cv.deinit();
    var count_matrix = try cv.fit_transform(allocator, corpus);
    defer count_matrix.deinit(allocator);

    std.debug.print("Count matrix ({} x {}):\n", .{ count_matrix.rows, count_matrix.cols });
    for (0..count_matrix.rows) |i| {
        std.debug.print("  doc{}: ", .{i});
        for (0..count_matrix.cols) |j| {
            const v = try count_matrix.get(i, j);
            std.debug.print("{d:.0} ", .{v});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\n", .{});

    // TF-IDF vectorizer
    var tfidf = zsl.ml.nlp.TfIdfVectorizer.init(allocator, 0);
    defer tfidf.deinit();
    var tfidf_matrix = try tfidf.fit_transform(allocator, corpus);
    defer tfidf_matrix.deinit(allocator);

    std.debug.print("TF-IDF matrix ({} x {}):\n", .{ tfidf_matrix.rows, tfidf_matrix.cols });
    for (0..tfidf_matrix.rows) |i| {
        std.debug.print("  doc{}: ", .{i});
        for (0..tfidf_matrix.cols) |j| {
            const v = try tfidf_matrix.get(i, j);
            std.debug.print("{d:.4} ", .{v});
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("\n", .{});

    // Lancaster stemming
    const words = &[_][]const u8{ "walking", "walked", "flies", "tried", "cats", "national" };
    std.debug.print("Stemmed words:\n", .{});
    for (words) |word| {
        const stemmed = try zsl.ml.nlp.stem(allocator, word);
        defer allocator.free(stemmed);
        std.debug.print("  {s} -> {s}\n", .{ word, stemmed });
    }
}
