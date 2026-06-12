const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zsl_mod = b.addModule("zsl", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });

    const test_step = b.step("test", "Run unit tests");
    const lib_tests = b.addTest(.{
        .root_module = zsl_mod,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_lib_tests.step);

    const example_step = b.step("example", "Run vector_ops example");
    const exe = b.addExecutable(.{
        .name = "vector_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/vector_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    const run_exe = b.addRunArtifact(exe);
    example_step.dependOn(&run_exe.step);
}
