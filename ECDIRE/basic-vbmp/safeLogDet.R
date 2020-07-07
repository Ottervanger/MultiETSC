`safeLogDet` <-
function(x) {
    safeLog(det(x));  # 2*sum(safeLog(diag(chol(x))))
}

