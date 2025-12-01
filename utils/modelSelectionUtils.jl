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
    crossValidationIndices::Vector{Int64},
    preprocessor::Function
)
    inputs, targets_raw = dataset
    targets = string.(targets_raw)   # para MLJ
    classes = unique(targets)
    numFolds = maximum(crossValidationIndices)

    # =========================
    # Caso ANN: ANNCrossValidation
    # =========================
    if modelType == :ANN
        topology          = get(modelHyperparameters, "topology", [5])
        transferFunctions = get(modelHyperparameters, "transferFunctions", fill(σ, length(topology)))
        learningRate      = get(modelHyperparameters, "learningRate", 0.01)
        validationRatio   = get(modelHyperparameters, "validationRatio", 0.2)
        numExecutions     = get(modelHyperparameters, "numExecutions", 50)
        maxEpochs         = get(modelHyperparameters, "maxEpochs", 1000)
        maxEpochsVal      = get(modelHyperparameters, "maxEpochsVal", 20)

        return ANNCrossValidation(
            topology, dataset, crossValidationIndices;
            numExecutions   = numExecutions,
            transferFunctions = transferFunctions,
            learningRate    = learningRate,
            validationRatio = validationRatio,
            maxEpochs       = maxEpochs,
            maxEpochsVal    = maxEpochsVal
        )
    end

    # =========================
    # Modelos clásicos (SVM, árbol, kNN)
    # =========================
    metrics_fold     = zeros(7, numFolds)
    globalConfMatrix = zeros(length(classes), length(classes))

    for fold in 1:numFolds
        # --- 1. Índices de train/valid del fold ---
        train_idx = findall(x -> x != fold, crossValidationIndices)
        val_idx   = findall(x -> x == fold, crossValidationIndices)

        # Datos crudos
        X_train_raw = inputs[train_idx, :]
        X_val_raw   = inputs[val_idx, :]
        y_train_raw = targets_raw[train_idx]   # tipo original (Bool, etc.)
        y_train     = targets[train_idx]       # string para MLJ
        y_val       = targets[val_idx]         # string para MLJ

        # --- 2. PREPROCESADO POR FOLD ---
        # Ajusta parámetros solo con X_train_raw y los aplica a train y valid
        X_train, X_val, _ = preprocessor(X_train_raw, X_val_raw, y_train_raw)

        # --- 3. Definir el modelo de MLJ ---
        model = nothing
        if modelType == :SVC
            kernel_str = get(modelHyperparameters, "kernel", "rbf")
            kernel = if kernel_str == "linear"
                LIBSVM.Kernel.Linear
            elseif kernel_str == "rbf"
                LIBSVM.Kernel.RadialBasis
            elseif kernel_str == "sigmoid"
                LIBSVM.Kernel.Sigmoid
            else
                LIBSVM.Kernel.Polynomial
            end
            cost   = Float64(get(modelHyperparameters, "C", 1.0))
            gamma  = Float64(get(modelHyperparameters, "gamma", 0.1))
            degree = Int32(get(modelHyperparameters, "degree", 3))
            coef0  = Float64(get(modelHyperparameters, "coef0", 0.0))
            model  = SVC_model(kernel=kernel, cost=cost, gamma=gamma,
                               degree=degree, coef0=coef0)

        elseif modelType == :DecisionTreeClassifier
            max_depth = get(modelHyperparameters, "max_depth", 4)
            model = DTC_model(max_depth=max_depth,
                              rng=Random.MersenneTwister(1))

        elseif modelType == :KNeighborsClassifier
            K = get(modelHyperparameters, "n_neighbors", 3)
            model = KNN_model(K=K)

        else
            error("Unsupported model: $modelType")
        end

        # --- 4. Entrenar y predecir ---
        X_train_tbl = MLJ.table(X_train)
        X_val_tbl   = MLJ.table(X_val)

        mach = machine(model, X_train_tbl, categorical(y_train))
        MLJ.fit!(mach, verbosity=0)
        y_hat = mode.(MLJ.predict(mach, X_val_tbl))

        # --- 5. Métricas en el fold ---
        results = confusionMatrix(y_hat, y_val, classes)
        metrics_fold[:, fold] .= results[1:7]
        globalConfMatrix      .+= results[8]
    end

    # Vector de medias por métrica (accuracy, error, sens, spec, ppv, npv, f1)
    metrics_mean = [mean(metrics_fold[i, :]) for i in 1:7]

    return metrics_mean, globalConfMatrix
end
