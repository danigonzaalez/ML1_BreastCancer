# models.jl
# Debe cargarse después de Approaches.jl (que define ModelSpec)

using .Approaches

# ANN
const MODEL_ANN = ModelSpec(
    :ANN,
    [  # 1 hidden layer
        Dict(:topology => [5],      :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [10],     :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [15],     :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [20],     :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [30],     :maxEpochs => 200, :learningRate => 0.01),
      # 2 hidden layers
        Dict(:topology => [10,5],  :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [20,10],  :maxEpochs => 200, :learningRate => 0.01),
        Dict(:topology => [30,15],  :maxEpochs => 200, :learningRate => 0.01)
    ]
)

# SVM
const MODEL_SVM = ModelSpec(
    :SVM,
    [   
        #liner with different C
        Dict(:kernel => :linear,    :C => 0.1),
        Dict(:kernel => :linear,    :C => 1.0),
        Dict(:kernel => :linear, :C => 10.0),

        #RBF with different C and gamma
        Dict(:kernel => :rbf,    :C => 0.1, :gamma => 0.01),
        Dict(:kernel => :rbf,    :C => 1.0, :gamma => 0.01),
        Dict(:kernel => :rbf,    :C => 0.5, :gamma => 0.1),

        #Polynomial
        Dict(:kernel => :poly,   :C => 1.0, :degree => 3),

        #Sigmoidal
        Dict(:kernel => :sigmoid,:C => 1.0, :gamma => 0.01)

    ]
)

# Árbol de decisión
const MODEL_TREE = ModelSpec(
    :DecisionTree,
    [
        Dict(:max_depth => 2),
        Dict(:max_depth => 3),
        Dict(:max_depth => 5),
        Dict(:max_depth => 6),
        Dict(:max_depth => 8),
        Dict(:max_depth => 10)
    ]
)

# kNN
const MODEL_KNN = ModelSpec(
    :kNN,
    [
        Dict(:K => 1),
        Dict(:K => 3),
        Dict(:K => 5),
        Dict(:K => 7),
        Dict(:K => 9),
        Dict(:K => 11)
    ]
)

const ALL_MODEL_SPECS = Dict(
    :ANN          => MODEL_ANN,
    :SVM          => MODEL_SVM,
    :DecisionTree => MODEL_TREE,
    :kNN          => MODEL_KNN
)
