pub const data = @import("ml/data.zig");
pub const workspace = @import("ml/workspace.zig");
pub const paramsreg = @import("ml/paramsreg.zig");
pub const linreg = @import("ml/linreg.zig");
pub const kmeans = @import("ml/kmeans.zig");
pub const knn = @import("ml/knn.zig");
pub const logreg = @import("ml/logreg.zig");
pub const svm = @import("ml/svm.zig");
pub const decision_tree = @import("ml/decision_tree.zig");
pub const lasso = @import("ml/lasso.zig");
pub const random_forest = @import("ml/random_forest.zig");
pub const nlp = @import("ml/nlp.zig");
pub const ridge = @import("ml/ridge.zig");
pub const elastic_net = @import("ml/elastic_net.zig");

test {
    _ = data;
    _ = workspace;
    _ = paramsreg;
    _ = linreg;
    _ = kmeans;
    _ = knn;
    _ = logreg;
    _ = svm;
    _ = decision_tree;
    _ = lasso;
    _ = random_forest;
    _ = nlp;
    _ = ridge;
    _ = elastic_net;
}
