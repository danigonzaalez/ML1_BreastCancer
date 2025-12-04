####################################
#         Ensemble Utils          #
####################################
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

function train_and_test_ensemble(
    estimators::AbstractArray{Symbol,1},
    modelsHyperParameters::AbstractArray{<:Dict,1},
    ensembleHyperParameters::Dict,
    X_train::AbstractArray{<:Real,2},
    y_train::AbstractArray{Bool,1},
    X_test::AbstractArray{<:Real,2},
    y_test::AbstractArray{Bool,1}
)
    # 1. Normalization (MinMax)
    # Replicating the logic from trainClassEnsemble: standardizes inputs using Train params
    normParams = Utils.calculateMinMaxNormalizationParameters(X_train)
    X_train_norm = Utils.normalizeMinMax(X_train, normParams)
    X_test_norm = Utils.normalizeMinMax(X_test, normParams)
    
    # 2. Data Conversion for MLJ
    X_train_df = DataFrame(X_train_norm, :auto)
    X_test_df  = DataFrame(X_test_norm, :auto)
    y_train_cat = categorical(string.(y_train)) # Categorical for MLJ
    
    # 3. Build Base Models
    # Dictionary to map symbols to loaded MLJ models (must match ensembleUtils.jl)
    # Note: These constants (SVC_model, etc.) are assumed to be loaded via Utils/ensembleUtils
    model_dict = Dict(
        :SVM => Utils.SVC_model,
        :DecisionTree => Utils.DTC_model,
        :kNN => Utils.KNN_model,
        :ANN => Utils.ANNClassifier
    )
    
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
    
    if isempty(base_models)
        error("No models created.")
    end
    
    # 4. Configure Ensemble
    voting_type = get(ensembleHyperParameters, :voting, :hard)
    ensemble_model = Utils.VotingClassifier(models = base_models, voting = voting_type)
    
    println("\nTraining Ensemble on full training set...")
    println("  - Voting strategy: ", voting_type)
    println("  - Number of models: ", length(base_models))
    
    # 5. Fit the Ensemble
    mach = machine(ensemble_model, X_train_df, y_train_cat)
    fit!(mach, verbosity=0)
    
    # 6. Predict on Test set
    y_pred_cat = predict_mode(mach, X_test_df)
    
    # Convert predictions back to Bool for evaluation (assuming "true" string is positive)
    y_pred = string.(y_pred_cat) .== "true"
    
    # 7. Evaluate
    println("\n======================================================================")
    println(" Final evaluation on Test Set (Ensemble)")
    println("======================================================================")
    Utils.printConfusionMatrix(y_pred, y_test)
    
    return y_pred
end
