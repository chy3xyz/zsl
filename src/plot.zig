pub const Plot = @import("plot/plot.zig").Plot;
pub const Layout = @import("plot/layout.zig").Layout;
pub const Axis = @import("plot/layout.zig").Axis;
pub const Annotation = @import("plot/layout.zig").Annotation;
pub const Marker = @import("plot/trace.zig").Marker;
pub const Line = @import("plot/trace.zig").Line;
pub const Trace = @import("plot/trace.zig").Trace;
pub const TraceType = @import("plot/trace.zig").TraceType;
pub const Mode = @import("plot/trace.zig").Mode;
pub const ScatterTrace = @import("plot/trace.zig").ScatterTrace;
pub const LineTrace = @import("plot/trace.zig").LineTrace;
pub const BarTrace = @import("plot/trace.zig").BarTrace;
pub const HeatmapTrace = @import("plot/trace.zig").HeatmapTrace;
pub const save_html = @import("plot/show.zig").save_html;
pub const show = @import("plot/show.zig").show;
pub const confusion_matrix = @import("plot/ml_plots.zig").confusion_matrix;
pub const roc_curve = @import("plot/ml_plots.zig").roc_curve;
pub const feature_importance = @import("plot/ml_plots.zig").feature_importance;

test {
    _ = @import("plot/plot.zig");
    _ = @import("plot/layout.zig");
    _ = @import("plot/trace.zig");
    _ = @import("plot/show.zig");
    _ = @import("plot/ml_plots.zig");
}
