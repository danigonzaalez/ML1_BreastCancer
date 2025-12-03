####################################
#         Ensemble Utils          #
####################################

# Aquí asumimos que en Utils.jl ya tienes:
# using MLJ
# using DataFrames
# using CategoricalArrays
# import LIBSVM, DecisionTree, NearestNeighborModels

# Y que en modelSelectionUtils.jl ya declaraste:
# const SVC_model = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
# const DTC_model = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
# const KNN_model = @load KNNClassifier pkg=NearestNeighborModels verbosity=0
#
# Si no es así, descomenta estas líneas:
# const SVC_model = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
# const DTC_model = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
# const KNN_model = @load KNNClassifier pkg=NearestNeighborModels verbosity=0
const ANNClassifier = @load NeuralNetworkClassifier pkg=MLJFlux

mutable struct VotingClassifier <: Probabilistic   # Models must be probabilistic, inherited from MLJBase
    models::Vector{Probabilistic}
    voting::Symbol  # :hard or :soft
    weights::Union{Nothing, Vector{Float64}}
end

function VotingClassifier(; models=Probabilistic[], voting=:hard, weights=nothing)
    @assert voting in [:hard, :soft] "The only possible labels are :hard or :soft"
    
    normalized_weights = nothing
    if weights !== nothing
        @assert length(weights) == length(models) "Number of weights must match number of models"
        @assert all(w >= 0 for w in weights) "All weights must be non-negative"
        
        # Normalize weights to sum to 1.0
        normalized_weights = Float64.(weights) ./ sum(weights)
    end
    
    return VotingClassifier(models, voting, normalized_weights)
end

function MLJModelInterface.fit(model::VotingClassifier, verbosity::Int, X, y)
    # Train each base model
    fitresults = []
    for base_model in model.models
        model_copy = deepcopy(base_model)
        mach = machine(model_copy, X, y)
        fit!(mach, verbosity=0)
        push!(fitresults, mach)
    end
    
    cache = nothing
    report = (n_models=length(model.models), voting=model.voting, weights=model.weights)
    
    return fitresults, cache, report
end

function MLJModelInterface.predict_mode(model::VotingClassifier, fitresult, Xnew)
    machines = fitresult
    
    # Get predictions from all base models
    predictions = [predict_mode(mach, Xnew) for mach in machines]
    
    # Convert predictions to String to avoid type conflicts
    predictions = [string.(p) for p in predictions]
    
    # Get all possible classes
    all_classes = unique(vcat([unique(p) for p in predictions]...))
    n_samples = length(predictions[1])
    n_models = length(machines)
    
    # Weights (default: equal)
    weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights
    
    # Output vector
    ensemble_pred = Vector{String}(undef, n_samples)
    
    # Weighted voting
    for i in 1:n_samples
        vote_counts = Dict(class => 0.0 for class in all_classes)
        for (j, pred) in enumerate(predictions)
            vote_counts[pred[i]] += weights[j]
        end
        ensemble_pred[i] = argmax(vote_counts)
    end
    
    return categorical(ensemble_pred)
end

function MLJModelInterface.predict(model::VotingClassifier, fitresult, Xnew)
    machines = fitresult
    
    result = if model.voting == :hard
        # For hard voting, return deterministic predictions
        UnivariateFinite(predict_mode(model, fitresult, Xnew))
    else
        # Soft voting: weighted average of probabilities
        all_predictions = [predict(mach, Xnew) for mach in machines]
        
        # Get class levels
        first_pred = all_predictions[1][1]
        class_levels = MLJBase.classes(first_pred)
        n_classes = length(class_levels)
        n_samples = length(all_predictions[1])
        n_models = length(machines)
        
        # Determine weights (equal if not specified)
        weights = model.weights === nothing ? fill(1.0/n_models, n_models) : model.weights
        
        # Weighted average of probabilities
        avg_probs = zeros(n_samples, n_classes)
        for (model_idx, preds) in enumerate(all_predictions)
            for i in 1:n_samples
                for (j, level) in enumerate(class_levels)
                    avg_probs[i, j] += weights[model_idx] * pdf(preds[i], level)
                end
            end
        end
        
        [UnivariateFinite(class_levels, avg_probs[i, :]) for i in 1:n_samples]
    end
    
    return result
end

MLJModelInterface.metadata_model(VotingClassifier,
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    load_path="VotingClassifier"
)


function trainClassEnsemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:Dict,1},
    ensembleHyperParameters::Dict,
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    kFoldIndices::Array{Int64,1}
)
    X_matrix, y_matrix = trainingDataset
    y_bool = vec(y_matrix)
    y = categorical(string.(y_bool)) # Categorical for MLJ
    
    @assert length(estimators) == length(modelsHyperParameters)
    k = length(unique(kFoldIndices))
    test_metrics = Vector{Float64}(undef, k)
    
    # Dictionary to map symbols to loaded MLJ models
    model_dict = Dict(
        :SVM => SVC_model,
        :DecisionTree => DTC_model,
        :kNN => KNN_model,
        :ANN => ANNClassifier
    )
    
    for numFold in 1:k
        # Split
        test_mask = kFoldIndices .== numFold
        train_mask = .!test_mask
        X_train_matrix = X_matrix[train_mask, :]
        X_test_matrix = X_matrix[test_mask, :]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Normalize
        normParams = calculateMinMaxNormalizationParameters(X_train_matrix)
        X_train_matrix = normalizeMinMax(X_train_matrix, normParams)
        X_test_matrix = normalizeMinMax(X_test_matrix, normParams)
        
        # DataFrame conversion for MLJ
        X_train = DataFrame(X_train_matrix, :auto)
        X_test = DataFrame(X_test_matrix, :auto)
        
        # Build Base Models
        base_models = []
        for (index, estimator) in enumerate(estimators)
            hyperparams = modelsHyperParameters[index]
            try
                if estimator == :SVM
                    push!(base_models, model_dict[:SVM]())
                elseif estimator == :DecisionTree
                    push!(base_models, model_dict[:DecisionTree](
                        max_depth = get(hyperparams, :max_depth, 5),
                        min_purity_increase = get(hyperparams, :min_purity_increase, 0.0)
                    ))
                elseif estimator == :kNN
                    push!(base_models, model_dict[:kNN](K = get(hyperparams, :K, 5)))
                elseif estimator == :ANN
                    topology = get(hyperparams, :topology, [10])
                    push!(base_models, model_dict[:ANN](
                        builder = MLJFlux.Short(
                            n_hidden = length(topology) > 0 ? topology[1] : 10,
                            dropout = 0.0, σ = Flux.relu
                        ),
                        epochs = get(hyperparams, :maxEpochs, 100),
                        optimiser = Flux.Adam(get(hyperparams, :learningRate, 0.01))
                    ))
                end
            catch e
                println("Error creating $estimator: $e")
            end
        end
        
        if isempty(base_models); error("No models created for fold $numFold"); end
        
        # Ensemble (Voting)
        voting_type = get(ensembleHyperParameters, :voting, :hard)
        ensemble_model = VotingClassifier(models = base_models, voting = voting_type)
        
        # Train & Eval
        try
            mach = machine(ensemble_model, X_train, y_train)
            fit!(mach, verbosity=0)
            y_pred = predict_mode(mach, X_test)
            metric = get(ensembleHyperParameters, :metric, accuracy)
            test_metrics[numFold] = metric(y_pred, y_test)
        catch e
            println("Error training ensemble: $e")
            test_metrics[numFold] = 0.0
        end
    end
    
    return (mean_metric = mean(test_metrics), std_metric = std(test_metrics), fold_metrics = test_metrics)
end