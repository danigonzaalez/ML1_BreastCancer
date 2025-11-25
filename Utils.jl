module Utils

###########################################################
##########      Machine Learning Utils Library   ##########
###########################################################

# --- Core Libraries ---
using Flux
using Flux.Losses
using Statistics
using LinearAlgebra
using Random
using Downloads

# --- MLJ & Ensembles Libraries ---
using MLJ
using MLJFlux
using DataFrames
using CategoricalArrays
import LIBSVM
import DecisionTree
import NearestNeighborModels

# --- Visualization ---
using Plots

# --- Submódulos lógicos (en realidad ficheros incluidos) ---
include("utils/preprocessingUtils.jl")
include("utils/ANNUtils.jl")
include("utils/metricsUtils.jl")
include("utils/modelSelectionUtils.jl")
include("utils/ensembleUtils.jl")
include("utils/IOandPlotUtils.jl")

# --- Exports principales (puedes ajustar) ---
export oneHotEncoding,
       calculateMinMaxNormalizationParameters,
       normalizeMinMax!, normalizeMinMax,
       classifyOutputs,
       holdOut, crossvalidation,
       buildClassANN, trainClassANN, ANNCrossValidation,
       confusionMatrix, printConfusionMatrix,
       modelCompilation,
       VotingClassifier, trainClassEnsemble,
       load_data, draw_results

end # module Utils
