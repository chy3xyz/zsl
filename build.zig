const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zsl_mod = b.addModule("zsl", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    zsl_mod.link_libc = true;

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

    const eigen_example_step = b.step("example-eigen", "Run eigen_ops example");
    const eigen_exe = b.addExecutable(.{
        .name = "eigen_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/eigen_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    eigen_example_step.dependOn(&b.addRunArtifact(eigen_exe).step);

    const statistics_example_step = b.step("example-statistics", "Run statistics_ops example");
    const statistics_exe = b.addExecutable(.{
        .name = "statistics_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/statistics_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    statistics_example_step.dependOn(&b.addRunArtifact(statistics_exe).step);

    const metrics_example_step = b.step("example-metrics", "Run metrics_ops example");
    const metrics_exe = b.addExecutable(.{
        .name = "metrics_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/metrics_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    metrics_example_step.dependOn(&b.addRunArtifact(metrics_exe).step);

    const roots_example_step = b.step("example-roots", "Run roots_ops example");
    const roots_exe = b.addExecutable(.{
        .name = "roots_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/roots_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    roots_example_step.dependOn(&b.addRunArtifact(roots_exe).step);

    const poly_example_step = b.step("example-poly", "Run poly_ops example");
    const poly_exe = b.addExecutable(.{
        .name = "poly_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/poly_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    poly_example_step.dependOn(&b.addRunArtifact(poly_exe).step);

    const prime_example_step = b.step("example-prime", "Run prime_ops example");
    const prime_exe = b.addExecutable(.{
        .name = "prime_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/prime_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    prime_example_step.dependOn(&b.addRunArtifact(prime_exe).step);

    const dist_example_step = b.step("example-dist", "Run dist_ops example");
    const dist_exe = b.addExecutable(.{
        .name = "dist_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/dist_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    dist_example_step.dependOn(&b.addRunArtifact(dist_exe).step);

    const iter_example_step = b.step("example-iter", "Run iter_ops example");
    const iter_exe = b.addExecutable(.{
        .name = "iter_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/iter_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    iter_example_step.dependOn(&b.addRunArtifact(iter_exe).step);

    const preprocessing_example_step = b.step("example-preprocessing", "Run preprocessing_ops example");
    const preprocessing_exe = b.addExecutable(.{
        .name = "preprocessing_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/preprocessing_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    preprocessing_example_step.dependOn(&b.addRunArtifact(preprocessing_exe).step);

    const model_selection_example_step = b.step("example-model-selection", "Run model_selection_ops example");
    const model_selection_exe = b.addExecutable(.{
        .name = "model_selection_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/model_selection_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    model_selection_example_step.dependOn(&b.addRunArtifact(model_selection_exe).step);

    const quaternion_example_step = b.step("example-quaternion", "Run quaternion_ops example");
    const quaternion_exe = b.addExecutable(.{
        .name = "quaternion_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/quaternion_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    quaternion_example_step.dependOn(&b.addRunArtifact(quaternion_exe).step);

    const noise_example_step = b.step("example-noise", "Run noise_ops example");
    const noise_exe = b.addExecutable(.{
        .name = "noise_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/noise_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    noise_example_step.dependOn(&b.addRunArtifact(noise_exe).step);

    const consts_example_step = b.step("example-consts", "Run consts_ops example");
    const consts_exe = b.addExecutable(.{
        .name = "consts_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/consts_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    consts_example_step.dependOn(&b.addRunArtifact(consts_exe).step);

    const fun_example_step = b.step("example-fun", "Run fun_ops example");
    const fun_exe = b.addExecutable(.{
        .name = "fun_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/fun_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    fun_example_step.dependOn(&b.addRunArtifact(fun_exe).step);

    const graph_example_step = b.step("example-graph", "Run graph_ops example");
    const graph_exe = b.addExecutable(.{
        .name = "graph_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/graph_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    graph_example_step.dependOn(&b.addRunArtifact(graph_exe).step);

    const gm_example_step = b.step("example-gm", "Run gm_ops example (Bins)");
    const gm_exe = b.addExecutable(.{
        .name = "gm_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/gm_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    gm_example_step.dependOn(&b.addRunArtifact(gm_exe).step);

    const gm_ps_example_step = b.step("example-gm-ps", "Run gm_point_segment_ops example");
    const gm_ps_exe = b.addExecutable(.{
        .name = "gm_point_segment_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/gm_point_segment_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    gm_ps_example_step.dependOn(&b.addRunArtifact(gm_ps_exe).step);

    const kmeans_example_step = b.step("example-kmeans", "Run kmeans_ops example");
    const kmeans_exe = b.addExecutable(.{
        .name = "kmeans_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/kmeans_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    kmeans_example_step.dependOn(&b.addRunArtifact(kmeans_exe).step);

    const linreg_example_step = b.step("example-linreg", "Run linreg_ops example");
    const linreg_exe = b.addExecutable(.{
        .name = "linreg_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/linreg_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    linreg_example_step.dependOn(&b.addRunArtifact(linreg_exe).step);

    const knn_example_step = b.step("example-knn", "Run knn_ops example");
    const knn_exe = b.addExecutable(.{
        .name = "knn_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/knn_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    knn_example_step.dependOn(&b.addRunArtifact(knn_exe).step);

    const deriv_example_step = b.step("example-deriv", "Run deriv_ops example");
    const deriv_exe = b.addExecutable(.{
        .name = "deriv_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/deriv_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    deriv_example_step.dependOn(&b.addRunArtifact(deriv_exe).step);

    const diff_example_step = b.step("example-diff", "Run diff_ops example");
    const diff_exe = b.addExecutable(.{
        .name = "diff_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/diff_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    diff_example_step.dependOn(&b.addRunArtifact(diff_exe).step);

    const fft_example_step = b.step("example-fft", "Run fft_ops example");
    const fft_exe = b.addExecutable(.{
        .name = "fft_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/fft_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    fft_example_step.dependOn(&b.addRunArtifact(fft_exe).step);

    const easings_example_step = b.step("example-easings", "Run easings_ops example");
    const easings_exe = b.addExecutable(.{
        .name = "easings_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/easings_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    easings_example_step.dependOn(&b.addRunArtifact(easings_exe).step);

    const csv_example_step = b.step("example-csv", "Run csv_ops example");
    const csv_exe = b.addExecutable(.{
        .name = "csv_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/csv_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    csv_example_step.dependOn(&b.addRunArtifact(csv_exe).step);

    const ml_advanced_example_step = b.step("example-ml-advanced", "Run ml_advanced_ops example");
    const ml_advanced_exe = b.addExecutable(.{
        .name = "ml_advanced_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/ml_advanced_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    ml_advanced_example_step.dependOn(&b.addRunArtifact(ml_advanced_exe).step);

    const plot_example_step = b.step("example-plot", "Run plot_ops example");
    const plot_exe = b.addExecutable(.{
        .name = "plot_ops",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/plot_ops.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zsl", .module = zsl_mod },
            },
        }),
    });
    plot_example_step.dependOn(&b.addRunArtifact(plot_exe).step);
}
