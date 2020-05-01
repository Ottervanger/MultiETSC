library("TSdist")

#Functions for GP training
source("code/fullearlytrainaccuracy.R")
source("code/GP.R")
source("code/distancematrix.R")
source("code/cvStratified.R")
source("code/load_data.R")
source("code/relGP.R")
source("code/crossvalidation.R")

#Functions for reliability 1 and 2
source("code/reliability.R")
source("code/reliability1.R")
source("code/reliability2.R")
source("code/checkrel2.R")


#modified vbmp functions
source("code/modified-vbmp/genCPP.binary.R")
source("code/modified-vbmp/predClass.R")
source("code/modified-vbmp/predictCPP.R")
source("code/modified-vbmp/tmean.binary.R")
source("code/modified-vbmp/vbmp.R")

#Other vbmp functions
source("code/basic-vbmp/printTrace.R")
source("code/basic-vbmp/computeKernel.R")
source("code/basic-vbmp/covParams.R")
source("code/basic-vbmp/distSqrd.R")
source("code/basic-vbmp/genCPP.classic.R")
source("code/basic-vbmp/gaussQuad.R")
source("code/basic-vbmp/genCPP.quad.R")
source("code/basic-vbmp/lowerBound.R")
source("code/basic-vbmp/plotDiagnostics.R")
source("code/basic-vbmp/predError.R")
source("code/basic-vbmp/predLik.R")
source("code/basic-vbmp/rexponential.R")
source("code/basic-vbmp/safeLog.R")
source("code/basic-vbmp/safeLogDet.R")
source("code/basic-vbmp/safeNormCDF.R")
source("code/basic-vbmp/safeNormPDF.R")
source("code/basic-vbmp/tmean.classic.R")
source("code/basic-vbmp/tmean.quad.R")
source("code/basic-vbmp/tmean.R")
source("code/basic-vbmp/varphiUpdate.R")

