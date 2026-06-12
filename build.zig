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

    const example_vector_step = b.step("example-vector", "Run vector_ops example");
    const vector_exe = b.addExecutable(.{
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
    example_vector_step.dependOn(&b.addRunArtifact(vector_exe).step);

    const example_step = b.step("example", "Run matrix_ops example");
    const matrix_exe = b.addExecutable(.{
        .name = "matrix_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/matrix_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    example_step.dependOn(&b.addRunArtifact(matrix_exe).step);

    const blas_example_step = b.step("example-blas", "Run blas_core example");
    const blas_exe = b.addExecutable(.{
        .name = "blas_core",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/blas_core.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    blas_example_step.dependOn(&b.addRunArtifact(blas_exe).step);

    const lapack_example_step = b.step("example-lapack", "Run lapack_solve example");
    const lapack_exe = b.addExecutable(.{
        .name = "lapack_solve",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/lapack_solve.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    lapack_example_step.dependOn(&b.addRunArtifact(lapack_exe).step);
}
