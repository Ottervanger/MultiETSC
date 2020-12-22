require("parallel")		# for explicit parallelism
require("RhpcBLASctl")	# for suppressing OpenBLAS parallelism
options(rgl.useNULL=TRUE);
require("TSdist") 		# Time Series distance lib
require("fastDistSqrd") # using a cpp implementation for better performance

#Functions for GP training
source("classAccuracy.R")
source("GP.R")
source("distancematrix.R")
source("cvStratified.R")
source("load_data.R")
source("relGP.R")
source("crossvalidation.R")

#Functions for reliability 1 and 2
source("reliability.R")
source("reliability1.R")
source("reliability2.R")
source("checkrel2.R")

#modified vbmp functions
source("modified-vbmp/genCPP.binary.R")
source("modified-vbmp/predClass.R")
source("modified-vbmp/predictCPP.R")
source("modified-vbmp/tmean.binary.R")
source("modified-vbmp/vbmp.R")

#Other vbmp functions
source("basic-vbmp/printTrace.R")
source("basic-vbmp/computeKernel.R")
source("basic-vbmp/covParams.R")
source("basic-vbmp/genCPP.classic.R")
source("basic-vbmp/gaussQuad.R")
source("basic-vbmp/genCPP.quad.R")
source("basic-vbmp/lowerBound.R")
source("basic-vbmp/plotDiagnostics.R")
source("basic-vbmp/predError.R")
source("basic-vbmp/predLik.R")
source("basic-vbmp/rexponential.R")
source("basic-vbmp/safeLog.R")
source("basic-vbmp/safeLogDet.R")
source("basic-vbmp/safeNormCDF.R")
source("basic-vbmp/safeNormPDF.R")
source("basic-vbmp/tmean.classic.R")
source("basic-vbmp/tmean.quad.R")
source("basic-vbmp/tmean.R")
source("basic-vbmp/varphiUpdate.R")

