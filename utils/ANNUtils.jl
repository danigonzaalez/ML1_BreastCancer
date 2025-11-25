#############################
#        ANN Utils         #
#############################

# Usa Flux y funciones de preprocessing ya cargadas en Utils.jl

# -------- Pérdidas en train / val / test --------

function calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
    (trainX, trainY) = trainingDataset
    (valX, valY) = validationDataset
    (testX, testY) = testDataset
    trainLoss = loss(ann, trainX', trainY')
    valLoss   = isempty(valX)  ? NaN : loss(ann, valX', valY')
    testLoss  = isempty(testX) ? NaN : loss(ann, testX', testY')
    return (trainLoss, valLoss, testLoss)
end

# -------- Construcción de la red --------

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

# -------- Entrenamiento de la ANN --------

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


# -------- Cross-validation específica para ANN --------

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