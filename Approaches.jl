module Approaches

using ..Utils
using MLJ
using Tables
using Random
using CSV
using DataFrames
import LIBSVM, DecisionTree, NearestNeighborModels
const PCA_model = @load PCA pkg=MultivariateStats verbosity=0
const ICA_model = @load ICA pkg=MultivariateStats verbosity=0
const LDA_model = @load LDA pkg=MultivariateStats verbosity=0


#############################
#      Tipos principales    #
#############################

"""
    ModelSpec

Describe un tipo de modelo y su grid de hiperparámetros.

- `name`       :: Symbol    (:ANN, :SVM, :DecisionTree, :kNN)
- `hyper_grid` :: Vector{Dict{Symbol,Any}}
"""
struct ModelSpec
    name::Symbol
    hyper_grid::Vector{Dict{Symbol,Any}}
end

"""
    ModelResult

- `best_params`      :: Dict{Symbol,Any}
- `best_metric`      :: Float64
- `metrics_history`  :: Vector{Float64}
"""
struct ModelResult
    best_params::Dict{Symbol,Any}
    best_metric::Float64
    metrics_history::Vector{Float64}
end

"""
    Approach

- `name`         :: String
- `preprocessor` :: Function  (Xtr, Xte, ytr) -> (Xtr_prep, Xte_prep, state)
- `model_specs`  :: Vector{ModelSpec}
"""
struct Approach
    name::String
    preprocessor::Function
    model_specs::Vector{ModelSpec}
end

"""
    ApproachResult

Resultado de ejecutar CV sobre un Approach.

- `approach`       :: Approach
- `model_results`  :: Dict{Symbol,ModelResult}
- `winner_name`    :: Symbol
- `winner_params`  :: Dict{Symbol,Any}
"""
struct ApproachResult
    approach::Approach
    model_results::Dict{Symbol,ModelResult}
    winner_name::Symbol
    winner_params::Dict{Symbol,Any}
end


#############################
#   Búsqueda de modelos CV  #
#############################
"""
    run_model_cv(spec, X, y, cv_indices, preprocessor)

Convierte hiperparámetros Symbol -> String y llama a Utils.modelCompilation,
que aplicará el preprocesado por fold.
"""
function run_model_cv(
    spec::ModelSpec,
    X::AbstractMatrix,
    y::AbstractVector,
    cv_indices::Vector{Int},
    preprocessor::Function
)
    best_metric = -Inf
    best_params = Dict{Symbol,Any}()
    history = Float64[]

    for params in spec.hyper_grid
        # Convertimos claves a String para modelCompilation
        hp_str = Dict{String,Any}()
        for (k,v) in params
            hp_str[string(k)] = v
        end

        modelType =
            spec.name == :ANN          ? :ANN :
            spec.name == :SVM          ? :SVC :
            spec.name == :DecisionTree ? :DecisionTreeClassifier :
            spec.name == :kNN          ? :KNeighborsClassifier :
            error("Unknown model name $(spec.name)")

        metrics, _ = Utils.modelCompilation(
            modelType,
            hp_str,
            (X, y),
            cv_indices,
            preprocessor
        )
        # metrics es Vector{Float64} (clásicos) o vector de tuplas (ANN)
        f2score = metrics[8] isa Tuple ? metrics[8][1] : metrics[8]
        push!(history, f2score)

        if f2score > best_metric
            best_metric = f2score
            best_params = params
        end
    end

    return ModelResult(best_params, best_metric, history)
end



#############################
#       Ejecutar Approach   #
#############################

function run_approach(
    approach::Approach,
    X_train::AbstractMatrix,
    y_train_vec::AbstractVector{Bool},
    cv_indices::Vector{Int}
)
    # 1. NO preprocesamos aquí. Se hará dentro de la CV por fold.

    # 2. CV por modelo, pasando el preprocesador del approach
    model_results = Dict{Symbol,ModelResult}()
    for spec in approach.model_specs
        res = run_model_cv(spec, X_train, y_train_vec, cv_indices, approach.preprocessor)
        model_results[spec.name] = res
    end

    # 3. Elegir ganador (solo modelos individuales)
    metrics = [model_results[s.name].best_metric for s in approach.model_specs]
    names   = [s.name for s in approach.model_specs]

    best_idx      = argmax(metrics)
    winner_name   = names[best_idx]
    winner_params = model_results[winner_name].best_params

    return ApproachResult(approach, model_results, winner_name, winner_params)
end



#############################
#  Entrenar ganador en test #
#############################

# Helpers: modelos MLJ para el entrenamiento final
const ProbSVC        = @load ProbabilisticSVC pkg=LIBSVM
const TreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
const KNNClassifier  = @load KNNClassifier pkg=NearestNeighborModels

"""
    build_mlj_model(winner::Symbol, best_params::Dict{Symbol,Any})

Devuelve una instancia del modelo MLJ correspondiente (SVM, árbol o kNN),
configurada con los mejores hiperparámetros.
"""
function build_mlj_model(winner::Symbol, best_params::Dict{Symbol,Any})
    if winner == :SVM
        return ProbSVC(
            kernel = LIBSVM.Kernel.RadialBasis,   # o podrías mapear :kernel si lo usas
            cost   = get(best_params, :C, 1.0),
            gamma  = get(best_params, :gamma, 0.1)
        )
    elseif winner == :DecisionTree
        return TreeClassifier(
            max_depth = get(best_params, :max_depth, 4)
        )
    elseif winner == :kNN
        return KNNClassifier(
            K = get(best_params, :K, 5)
        )
    else
        error("build_mlj_model: modelo ganador no soportado: $winner")
    end
end

"""
    train_and_evaluate_winner(result, X_train, y_train_vec, X_test, y_test_vec)

Aplica el preprocesador del approach, entrena el modelo ganador con TODO el train
y evalúa en test (con una única función, independientemente del modelo, salvo ANN).
"""
function train_and_evaluate_winner(
    result::ApproachResult,
    X_train::AbstractMatrix, y_train_vec::AbstractVector{Bool},
    X_test::AbstractMatrix,  y_test_vec::AbstractVector{Bool};
    rng = Random.default_rng()
)
    approach = result.approach

    # 1) Preprocesado completo train/test con el preprocesador del approach
    Xtr_prep, Xte_prep, _ = approach.preprocessor(X_train, X_test, y_train_vec)

    winner      = result.winner_name
    best_params = result.model_results[winner].best_params

    # 2) Caso especial: ANN (usa tu entreno propio)
    if winner == :ANN
        classes = unique(y_train_vec)
        Ytr = Utils.oneHotEncoding(y_train_vec, classes)

        # small internal holdout dentro del train
        N   = size(Xtr_prep, 1)
        idx = collect(1:N)
        Random.shuffle!(rng, idx)
        n_val   = max(1, round(Int, 0.2N))
        val_idx = idx[1:n_val]
        tr_idx  = idx[n_val+1:end]

        X_tr  = Xtr_prep[tr_idx, :]
        Y_tr  = Ytr[tr_idx, :]
        X_val = Xtr_prep[val_idx, :]
        Y_val = Ytr[val_idx, :]

        topology     = get(best_params, :topology, [10])
        maxEpochs    = get(best_params, :maxEpochs, 200)
        learningRate = get(best_params, :learningRate, 0.01)

        ann, _, _, _ = Utils.trainClassANN(
            topology,
            (X_tr, Y_tr);
            validationDataset = (X_val, Y_val),
            testDataset       = (Array{Float64}(undef,0,0), falses(0,size(Ytr,2))),
            maxEpochs         = maxEpochs,
            learningRate      = learningRate
        )

        # Importante para evitar warning de Flux: entrada en Float32
        Xte32    = Float32.(Xte_prep)
        y_scores = ann(Xte32')'
        y_bin    = Utils.classifyOutputs(y_scores)
        yhat_bool = vec(y_bin[:, 1])

        Utils.printConfusionMatrix(yhat_bool, y_test_vec)
        return yhat_bool
    end

    # 3) Caso general: modelos MLJ (SVM, árbol, kNN) -> único flujo
    model = build_mlj_model(winner, best_params)

    Xtr_df  = DataFrame(Xtr_prep, :auto)
    Xte_df  = DataFrame(Xte_prep, :auto)
    ytr_cat = categorical(y_train_vec)
    yte_cat = categorical(y_test_vec)

    mach = machine(model, Xtr_df, ytr_cat, scitype_check_level=0)
    fit!(mach, verbosity=0)

    yhat_mode = mode.(predict(mach, Xte_df))
    yhat_bool = yhat_mode .== true

    Utils.printConfusionMatrix(yhat_bool, y_test_vec)
    return yhat_bool
end



export ModelSpec, ModelResult, Approach, ApproachResult,
       run_approach, train_and_evaluate_winner

end # module Approaches
