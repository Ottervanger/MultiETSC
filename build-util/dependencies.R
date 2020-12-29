pkgs = c("parallel", "TSdist", "RhpcBLASctl", "fastDistSqrd", "R.utils", "utils", "pso", "GA", "GenSA")
toInstall = c()

for (pkg in pkgs) {
    if (!(pkg %in% rownames(installed.packages()))) {
        cat(sprintf("Missing package: %s\n", pkg))
        toInstall = c(toInstall, pkg)
    }
}

if (length(toInstall)) {
    cat("The following packages are missing:\n")
    for (pkg in toInstall) {
        cat(pkg, '\n')
    }
    cat("Do you want to install the missing packages? [Y]/n: ")
    res = readLines("stdin",n=1)
    if (res != '' && tolower(substring(res, 1, 1)) != 'y') {
        stop(sprintf("Missing %d required R package(s)", length(toInstall)))
    }
    ## Default repo
    local({r <- getOption("repos")
           r["CRAN"] <- "https://cloud.r-project.org" 
           options(repos=r)
    })
    for (pkg in toInstall) {
        if (pkg == 'fastDistSqrd') {
            install.packages('algorithms/ECDIRE/Rcpp/fastDistSqrd_1.0.tar.gz', type="source")
        } else {
            install.packages(pkg)
        }
    }
}