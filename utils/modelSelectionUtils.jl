####################################
#     Model Selection (CV) Utils   #
####################################

# Carga de modelos base MLJ
const SVC_model = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
const DTC_model = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
const KNN_model = @load KNNClassifier pkg=NearestNeighborModels verbosity=0

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(collect(1:k), ceil(Int, N / k))[1:N]
    shuffle!(indices)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = Array{Int64,1}(undef, length(targets))
    indices[findall(targets)] = crossvalidation(sum(targets), k)
    indices[findall(.!targets)] = crossvalidation(sum(.!targets), k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = Array{Int64,1}(undef, size(targets,1))
    for i in 1:size(targets,2)
        n = sum(targets[:, i])
        @assert n >= k "Class $i has fewer instances ($n) than folds ($k)."
        indices[targets[:, i]] = crossvalidation(n, k)
    end
    return indices
end

crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64) = crossvalidation(oneHotEncoding(targets), k)

function modelCompilation(
        modelType::Symbol, 
        modelHyperparameters::Dict,
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Vector{Int64}
    )

    inputs, targets = dataset
    targets = string.(targets) # Convert targets to strings for MLJ
    classes = unique(targets)
    numFolds = maximum(crossValidationIndices)

    # ANN case: call ANNCrossValidation
    if modelType == :ANN
        topology = get(modelHyperparameters, "topology", [5])
        transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology)))
        learningRate = get(modelHyperparameters, "learningRate", 0.01)
        validationRatio = get(modelHyperparameters, "validationRatio", 0.2)
        numExecutions = get(modelHyperparameters, "numExecutions", 50)
        maxEpochs = get(modelHyperparameters, "maxEpochs", 1000)
        maxEpochsVal = get(modelHyperparameters, "maxEpochsVal", 20)

        return ANNCrossValidation(
            topology, dataset, crossValidationIndices;
            numExecutions=numExecutions, transferFunctions=transferFunctions,
            learningRate=learningRate, validationRatio=validationRatio,
            maxEpochs=maxEpochs, maxEpochsVal=maxEpochsVal
        )
    end

    # Vectors for metrics
    metrics_fold = zeros(7, numFolds)
    globalConfMatrix = zeros(length(classes), length(classes))

    for fold in 1:numFolds
        train_idx = findall(x -> x != fold, crossValidationIndices)
        test_idx = findall(x -> x == fold, crossValidationIndices)

        X_train, y_train = inputs[train_idx, :], targets[train_idx]
        X_test, y_test = inputs[test_idx, :], targets[test_idx]

        # Normalization (Unit 6 Correction: Only use training data for params)
        normParams = calculateMinMaxNormalizationParameters(X_train)
        normalizeMinMax!(X_train, normParams)
        normalizeMinMax!(X_test, normParams)

        # Define Model
        model = nothing
        if modelType == :SVC
            kernel_str = get(modelHyperparameters, "kernel", "rbf")
            kernel = if kernel_str == "linear"; LIBSVM.Kernel.Linear
                    elseif kernel_str == "rbf"; LIBSVM.Kernel.RadialBasis
                    elseif kernel_str == "sigmoid"; LIBSVM.Kernel.Sigmoid
                    else; LIBSVM.Kernel.Polynomial; end
            cost = Float64(get(modelHyperparameters, "C", 1.0))
            gamma = Float64(get(modelHyperparameters, "gamma", 0.1))
            degree = Int32(get(modelHyperparameters, "degree", 3))
            coef0 = Float64(get(modelHyperparameters, "coef0", 0.0))
            model = SVC_model(kernel=kernel, cost=cost, gamma=gamma, degree=degree, coef0=coef0)
        elseif modelType == :DecisionTreeClassifier
            max_depth = get(modelHyperparameters, "max_depth", 4)
            model = DTC_model(max_depth=max_depth, rng=Random.MersenneTwister(1))
        elseif modelType == :KNeighborsClassifier
            K = get(modelHyperparameters, "n_neighbors", 3)
            model = KNN_model(K=K)
        else
            error("Unsupported model: $modelType")
        end

        # Train & Predict
        mach = machine(model, MLJ.table(X_train), categorical(y_train))
        MLJ.fit!(mach, verbosity=0)
        
        y_hat = mode.(MLJ.predict(mach, MLJ.table(X_test)))
        
        # Eval
        results = confusionMatrix(y_hat, y_test, classes)
        metrics_fold[:, fold] .= results[1:7]
        globalConfMatrix .+= results[8]
    end

    return [ (mean(metrics_fold[i,:]), std(metrics_fold[i,:])) for i in 1:7 ]..., globalConfMatrix
end