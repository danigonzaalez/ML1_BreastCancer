module Utils

    ###########################################################
    ##########      Machine Learning Utils Library   ##########
    ###########################################################

    # --- Core Libraries ---
    using Flux
    using Flux.Losses
    using Statistics
    using DelimitedFiles
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

    # --------------------------------------------------------
    # --------------- Unit 2: One-hot Encoding ---------------
    # --------------------------------------------------------

    function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
        @assert(all([in(value, classes) for value in feature]))
        numClasses = length(classes)
        @assert(numClasses > 1)
        
        if (numClasses == 2)
            oneHot = reshape(feature .== classes[1], :, 1)
        else
            oneHot = BitArray{2}(undef, length(feature), numClasses)
            for numClass = 1:numClasses
                oneHot[:, numClass] .= (feature .== classes[numClass])
            end
        end
        return oneHot
    end

    oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))
    oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)


    # ---------------------------------------------------------
    # -------------- Unit 2: MinMax Normalization -------------
    # ---------------------------------------------------------

    function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
        return minimum(dataset, dims=1), maximum(dataset, dims=1)
    end

    function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, 
            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
        minValues = normalizationParameters[1]
        maxValues = normalizationParameters[2]
        dataset .-= minValues
        dataset ./= (maxValues .- minValues)
        dataset[:, vec(minValues .== maxValues)] .= 0
        return dataset
    end

    function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
        normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
    end

    function normalizeMinMax(dataset::AbstractArray{<:Real,2}, 
            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
        normalizeMinMax!(copy(dataset), normalizationParameters)
    end

    function normalizeMinMax(dataset::AbstractArray{<:Real,2})
        normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset))
    end


    # --------------------------------------------------
    # ----------- Unit 2: Classify Outputs -------------
    # --------------------------------------------------

    function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5) 
        numOutputs = size(outputs, 2)
        @assert(numOutputs != 2)
        if numOutputs == 1
            return outputs .>= threshold
        else
            (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
            outputs = falses(size(outputs))
            outputs[indicesMaxEachInstance] .= true
            @assert(all(sum(outputs, dims=2) .== 1))
            return outputs
        end
    end


    # ---------------------------------------------------------
    # -------------- Unit 3: Training Functions ---------------
    # ---------------------------------------------------------

    function holdOut(N::Int, P::Real)
        @assert 0.0 <= P <= 1.0
        @assert N > 1
        indices = randperm(N)
        n_test = floor(Int, P * N)
        n_train = N - n_test
        train_idx = n_train > 0 ? indices[1:n_train] : Int[]
        test_idx  = n_test > 0  ? indices[(n_train+1):end] : Int[]
        return train_idx, test_idx
    end

    function holdOut(N::Int, Pval::Real, Ptest::Real)
        train_val_idx, test_idx = holdOut(N, Ptest)
        N_train_val = length(train_val_idx)
        train_idx = Int[]
        val_idx = Int[]
        if N_train_val > 0 && Pval > 0
            Pval_rel = (1 - Ptest) > 0 ? Pval / (1 - Ptest) : 0.0
            Pval_rel = clamp(Pval_rel, 0.0, 1.0)
            train_idx_sub, val_idx_sub = holdOut(N_train_val, Pval_rel)
            train_idx = train_val_idx[train_idx_sub]
            val_idx   = train_val_idx[val_idx_sub]
        else
            train_idx = train_val_idx
        end
        return train_idx, val_idx, test_idx
    end

    function calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
        (trainX, trainY) = trainingDataset
        (valX, valY) = validationDataset
        (testX, testY) = testDataset
        trainLoss = loss(ann, trainX', trainY')
        valLoss   = isempty(valX)  ? NaN : loss(ann, valX', valY')
        testLoss  = isempty(testX) ? NaN : loss(ann, testX', testY')
        return (trainLoss, valLoss, testLoss)
    end

    function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
        ann = Chain()
        numInputsLayer = numInputs
        for numHiddenLayer in 1:length(topology)
            numNeurons = topology[numHiddenLayer]
            ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]))
            numInputsLayer = numNeurons
        end
        if (numOutputs == 1)
            ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
        else
            ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
            ann = Chain(ann..., softmax)
        end
        return ann
    end    

    function trainClassANN(topology::AbstractArray{<:Int,1},  
                trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
                validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
                transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
                maxEpochsVal::Int=20, showText::Bool=false) 

        (trainX, trainY) = trainingDataset
        (valX, valY) = validationDataset
        (testX, testY) = testDataset

        ann = buildClassANN(size(trainX,2), topology, size(trainY,2); transferFunctions=transferFunctions)
        loss(model, x, y) = size(y,1) == 1 ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

        trainLossHistory, valLossHistory, testLossHistory = Float64[], Float64[], Float64[]
        bestValidationLoss = Inf
        numEpochsWithoutImprovement = 0
        numEpoch = 0
        bestANN = deepcopy(ann)

        trainLoss, valLoss, testLoss = calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
        push!(trainLossHistory, trainLoss)
        push!(valLossHistory, valLoss)
        push!(testLossHistory, testLoss)

        opt_state = Flux.setup(Adam(learningRate), ann)

        while (numEpoch < maxEpochs) && (trainLoss > minLoss) && (numEpochsWithoutImprovement < maxEpochsVal)
            Flux.train!(loss, ann, [(trainX', trainY')], opt_state)
            numEpoch += 1
            trainLoss, valLoss, testLoss = calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
            push!(trainLossHistory, trainLoss)
            push!(valLossHistory, valLoss)
            push!(testLossHistory, testLoss)

            if !isempty(valX)
                if valLoss < bestValidationLoss
                    bestValidationLoss = valLoss
                    bestANN = deepcopy(ann)
                    numEpochsWithoutImprovement = 0
                else
                    numEpochsWithoutImprovement += 1
                end
            end

            if showText && numEpoch % 10 == 0 
                println("Epoch: $numEpoch | TrainLoss=$trainLoss | ValLoss=$valLoss | TestLoss=$testLoss")
            end
        end
        return (isempty(valX) ? ann : bestANN), trainLossHistory, valLossHistory, testLossHistory
    end

    function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false)

        trainX, trainY = trainingDataset
        valX, valY = validationDataset
        testX, testY = testDataset
        return trainClassANN(topology, (trainX, reshape(trainY,:,1)), 
            validationDataset=(valX, reshape(valY,:,1)), 
            testDataset=(testX, reshape(testY,:,1)), 
            transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, 
            learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText)
    end


    # ---------------------------------------------------------
    # -------------- Unit 4: Confusion Matrices ---------------
    # ---------------------------------------------------------

    function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
        @assert length(outputs) == length(targets)
        TP = sum(outputs .& targets)
        TN = sum(.!outputs .& .!targets)
        FP = sum(outputs .& .!targets)
        FN = sum(.!outputs .& targets)
        total = TP + TN + FP + FN
        
        accuracy = total == 0 ? 0.0 : (TP + TN) / total
        error_rate = total == 0 ? 0.0 : (FP + FN) / total
        sensitivity = (TP + FN) == 0 ? (TN == total ? 1.0 : 0.0) : TP / (TP + FN)
        specificity = (TN + FP) == 0 ? (TP == total ? 1.0 : 0.0) : TN / (TN + FP)
        ppv = (TP + FP) == 0 ? (TN == total ? 1.0 : 0.0) : TP / (TP + FP)
        npv = (TN + FN) == 0 ? (TP == total ? 1.0 : 0.0) : TN / (TN + FN)
        fscore = (sensitivity + ppv) == 0 ? 0.0 : 2 * (sensitivity * ppv) / (sensitivity + ppv)
        
        return accuracy, error_rate, sensitivity, specificity, ppv, npv, fscore, [TN FP; FN TP]
    end

    function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
        return confusionMatrix(outputs .>= threshold, targets)
    end

    function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=false)
        numClasses = size(outputs, 2)
        if numClasses == 1; return confusionMatrix(vec(outputs), vec(targets)); end
        if numClasses == 2; error("Two-column matrices are invalid (redundant binary case)."); end

        metrics = zeros(5, numClasses) # sens, spec, ppv, npv, f1
        validClasses = falses(numClasses)

        for i in 1:numClasses
            if sum(targets[:, i]) > 0
                _, _, metrics[1,i], metrics[2,i], metrics[3,i], metrics[4,i], metrics[5,i], _ = confusionMatrix(outputs[:, i], targets[:, i])
                validClasses[i] = true
            end
        end

        confMatrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:numClasses, j in 1:numClasses]
        
        if weighted
            counts = vec(sum(targets, dims=1)); total = sum(counts)
            finalMetrics = [sum(counts .* metrics[r,:]) / total for r in 1:5]
        else
            valid_idx = findall(validClasses); n_valid = length(valid_idx)
            finalMetrics = n_valid > 0 ? [sum(metrics[r,valid_idx]) / n_valid for r in 1:5] : fill(NaN, 5)
        end
        
        accuracy = sum(diag(confMatrix)) / sum(confMatrix)
        return accuracy, 1-accuracy, finalMetrics[1], finalMetrics[2], finalMetrics[3], finalMetrics[4], finalMetrics[5], confMatrix
    end

    function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
        return confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)
    end

    function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
        return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted)
    end

    function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
        return confusionMatrix(outputs, targets, unique(vcat(targets, outputs)); weighted=weighted)
    end

    # --- Print Functions ---
    function printConfusionMatrix(args...; kwargs...)
        results = confusionMatrix(args...; kwargs...)
        confMat = results[end]
        acc = results[1]
        println("Confusion Matrix:\n$confMat\n")
        println("Accuracy: $acc")
        println("Error rate: $(results[2])")
        println("Sensitivity: $(results[3])")
        println("Specificity: $(results[4])")
        println("PPV: $(results[5])")
        println("NPV: $(results[6])")
        println("F1-score: $(results[7])")
    end


    # ---------------------------------------------------------
    # -------------- Unit 5: Cross Validation -----------------
    # ---------------------------------------------------------

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

    function ANNCrossValidation(topology::AbstractArray{<:Int,1},
            dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
            crossValidationIndices::Array{Int64,1};
            numExecutions::Int=50,
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
            validationRatio::Real=0, maxEpochsVal::Int=20)
        
        inputs, targets = dataset
        classes = unique(targets)
        outputs = oneHotEncoding(targets, classes)
        numFolds = maximum(crossValidationIndices)
        
        metrics_fold = zeros(7, numFolds) # acc, err, sens, spec, ppv, npv, f1
        globalConfMatrix = zeros(length(classes), length(classes))

        for fold in 1:numFolds
            train_idx = findall(x -> x != fold, crossValidationIndices)
            test_idx = findall(x -> x == fold, crossValidationIndices)
            X_train, Y_train = inputs[train_idx, :], outputs[train_idx, :]
            X_test, Y_test = inputs[test_idx, :], outputs[test_idx, :]

            normParams = calculateMinMaxNormalizationParameters(X_train)
            normalizeMinMax!(X_train, normParams)
            normalizeMinMax!(X_test, normParams)

            metrics_rep = zeros(7, numExecutions)
            confMat_fold = zeros(length(classes), length(classes))

            for rep in 1:numExecutions
                if validationRatio > 0
                    t_sub, v_idx = holdOut(size(X_train,1), validationRatio)
                    X_t, Y_t = X_train[t_sub, :], Y_train[t_sub, :]
                    X_v, Y_v = X_train[v_idx, :], Y_train[v_idx, :]
                else
                    X_t, Y_t, X_v, Y_v = X_train, Y_train, Array{Float64}(undef,0,0), falses(0, length(classes))
                end
                
                ann, _, _, _ = trainClassANN(topology, (X_t, Y_t), validationDataset=(X_v, Y_v), 
                    testDataset=(X_test, Y_test), transferFunctions=transferFunctions, 
                    maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal)
                
                results = confusionMatrix(ann(X_test')', Y_test)
                metrics_rep[:, rep] .= results[1:7]
                confMat_fold .+= results[8]
            end
            metrics_fold[:, fold] = mean(metrics_rep, dims=2)
            globalConfMatrix .+= (confMat_fold ./ numExecutions)
        end
        
        return [ (mean(metrics_fold[i,:]), std(metrics_fold[i,:])) for i in 1:7 ]..., globalConfMatrix
    end


    # ---------------------------------------------------------
    # -------------- Unit 6: Model Compilation ----------------
    # ---------------------------------------------------------

    # Load models for compilation
    SVC_model = @load ProbabilisticSVC pkg=LIBSVM verbosity=0
    DTC_model = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0
    KNN_model = @load KNNClassifier pkg=NearestNeighborModels verbosity=0

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
            
            y_hat = modelType == :SVC ? MLJ.predict(mach, MLJ.table(X_test)) : mode.(MLJ.predict(mach, MLJ.table(X_test)))
            
            # Eval
            results = confusionMatrix(y_hat, y_test, classes)
            metrics_fold[:, fold] .= results[1:7]
            globalConfMatrix .+= results[8]
        end

        return [ (mean(metrics_fold[i,:]), std(metrics_fold[i,:])) for i in 1:7 ]..., globalConfMatrix
    end


    # ---------------------------------------------------------
    # -------------- Unit 7: Ensembles ------------------------
    # ---------------------------------------------------------

    # Load NeuralNetworkClassifier for Ensemble
    ANNClassifier = MLJ.@load NeuralNetworkClassifier pkg=MLJFlux verbosity=0

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


    # ---------------------------------------------------------
    # -------------- Unit 8: Data Loading & Plots -------------
    # ---------------------------------------------------------

    function load_data(filename::String, url::String)
        if !isfile(filename)
            Downloads.download(url, filename)
        end
        data = readdlm(filename, ',')
        return data
    end

    function draw_results(x, y; colors, target_names=nothing)
        num_classes = length(unique(colors))

        if !isnothing(target_names)
            @assert num_classes == length(target_names)
            label = target_names
        else
            label = [string(i) for i in 1:num_classes]
        end

        fig = plot()
        if (num_classes == 2)
            # Assuming y is one-hot or boolean
            possitive_class = y[:,1] .== 1
            scatter!(fig, x[possitive_class,1], x[possitive_class,2], markercolor=colors[1], label=label[1])
            scatter!(fig, x[.!possitive_class,1], x[.!possitive_class,2], markercolor=colors[2], label=label[2])
        else
            for i in 1:num_classes
                index_class = y[:,i] .== 1
                scatter!(fig, x[index_class, 1], x[index_class, 2], markercolor=colors[i], label=label[i])
            end
        end
        return fig
    end
end