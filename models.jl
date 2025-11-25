# models.jl
# Debe cargarse después de Approaches.jl (que define ModelSpec)

using .Approaches

# ANN
const MODEL_ANN = ModelSpec(
    :ANN,
    [
        Dict(:topology => [5],      :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [10],     :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [20],     :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [10,10],  :maxEpochs => 200, :learningRate => 0.01)
    ]
)

# SVM
const MODEL_SVM = ModelSpec(
    :SVM,
    [
        Dict(:kernel => :rbf,    :C => 1.0),
        Dict(:kernel => :rbf,    :C => 10.0),
        Dict(:kernel => :linear, :C => 1.0),
    ]
)

# Árbol de decisión
const MODEL_TREE = ModelSpec(
    :DecisionTree,
    [
        Dict(:max_depth => 3),
        Dict(:max_depth => 5),
        Dict(:max_depth => 10),
    ]
)

# kNN
const MODEL_KNN = ModelSpec(
    :kNN,
    [
        Dict(:K => 3),
        Dict(:K => 5),
        Dict(:K => 7),
    ]
)

const ALL_MODEL_SPECS = Dict(
    :ANN          => MODEL_ANN,
    :SVM          => MODEL_SVM,
    :DecisionTree => MODEL_TREE,
    :kNN          => MODEL_KNN
)
