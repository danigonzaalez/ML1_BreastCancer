###########################################################
##########      Machine Learning Utils Library   ##########
###########################################################

using Flux
using Flux.Losses
using Statistics
using DelimitedFiles
using LinearAlgebra
using Random

# --------------------------------------------------------
# --------------- Unit 2: One-hot Encoding ---------------
# --------------------------------------------------------

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]))
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes)
    @assert(numClasses > 1)
    
    if (numClasses == 2)
        # Case with only two classes
        oneHot = reshape(feature .== classes[1], :, 1)
    else
        # Case with more than two classes
        oneHot = BitArray{2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
    end
    return oneHot
end

# Overload of the OneHotEncoding method (auto-detect classes)
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# Overload of the OneHotEncoding method (boolean)
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1)


# ---------------------------------------------------------
# -------------- Unit 2: MinMax Normalization -------------
# ---------------------------------------------------------

# Calculate Min-Max normalization parameters
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end

# Min-Max Normalization (In-place with parameters)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, 
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1]
    maxValues = normalizationParameters[2]
    dataset .-= minValues
    dataset ./= (maxValues .- minValues)
    # eliminate any attribute that do not add information (avoid NaN division by zero)
    dataset[:, vec(minValues .== maxValues)] .= 0
    return dataset
end

# Min-Max Normalization overload (In-place, calculate parameters)
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset, calculateMinMaxNormalizationParameters(dataset))
end

# Min-Max Normalization overload (Copy with parameters)
function normalizeMinMax(dataset::AbstractArray{<:Real,2}, 
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    normalizeMinMax!(copy(dataset), normalizationParameters)
end

# Min-Max Normalization overload (Copy, calculate parameters)
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
        # Look for the maximum value using the findmax function
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        # Set up the boolean matrix to everything false while max values are true
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2) .== 1))
        return outputs
    end
end


# ---------------------------------------------------------
# -------------- Unit 3: Training Functions ---------------
# ---------------------------------------------------------

# HoldOut function (Train/Test)
function holdOut(N::Int, P::Real)
    # Validate inputs
    @assert 0.0 <= P <= 1.0 "P must be between 0 and 1"
    @assert N > 1 "N must be greater than 1"

    # Random permutation of indices 1..N
    indices = randperm(N)

    # Number of test samples (floored to avoid exceeding N)
    n_test = floor(Int, P * N)
    n_train = N - n_test

    # Handle empty cases safely
    train_idx = n_train > 0 ? indices[1:n_train] : Int[]
    test_idx  = n_test > 0  ? indices[(n_train+1):end] : Int[]

    return train_idx, test_idx
end

# Function that splits N patterns into three sets: training, validation, and test
function holdOut(N::Int, Pval::Real, Ptest::Real)
    # Check inputs
    @assert 0.0 <= Pval <= 1.0 "Pval must be between 0 and 1"
    @assert 0.0 <= Ptest <= 1.0 "Ptest must be between 0 and 1"
    @assert Pval + Ptest <= 1.0 "The sum of Pval and Ptest must be less than or equal to 1"
    @assert N > 1 "N must be greater than 1"

    # First split: separate the test set
    train_val_idx, test_idx = holdOut(N, Ptest)
    N_train_val = length(train_val_idx)

    # Initialize outputs
    train_idx = Int[]
    val_idx = Int[]

    # If there are samples left and Pval > 0, split them into train/validation
    if N_train_val > 0 && Pval > 0
        # Compute relative validation fraction within the remaining (non-test) set
        Pval_rel = (1 - Ptest) > 0 ? Pval / (1 - Ptest) : 0.0
        Pval_rel = clamp(Pval_rel, 0.0, 1.0)

        train_idx_sub, val_idx_sub = holdOut(N_train_val, Pval_rel)

        # Map relative indices back to the original
        train_idx = train_val_idx[train_idx_sub]
        val_idx   = train_val_idx[val_idx_sub]
    else
        # If Pval = 0 or no samples left (all were assigned to test)
        train_idx = train_val_idx
    end

    return train_idx, val_idx, test_idx
end

# Function that calculates train, validation and test losses
function calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
    (trainX, trainY) = trainingDataset
    (valX, valY) = validationDataset
    (testX, testY) = testDataset

    trainLoss = loss(ann, trainX', trainY')
    valLoss   = isempty(valX)  ? NaN : loss(ann, valX', valY')
    testLoss  = isempty(testX) ? NaN : loss(ann, testX', testY')

    return (trainLoss, valLoss, testLoss)
end

# Function that builds the ANN
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

# Function that trains the ANN
function trainClassANN(topology::AbstractArray{<:Int,1},  
            trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
            validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)), 
            transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
            maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
            maxEpochsVal::Int=20, showText::Bool=false) 

    # --- Load data ---
    (trainX, trainY) = trainingDataset
    (valX, valY) = validationDataset
    (testX, testY) = testDataset

    # --- Checks ---
    @assert size(trainX,1) == size(trainY,1) "Train inputs and targets mismatch"
    @assert size(valX,1) == size(valY,1) "Validation inputs and targets mismatch"
    @assert size(testX,1) == size(testY,1) "Test inputs and targets mismatch"
    !isempty(valX)  && @assert size(trainX,2) == size(valX,2)   "Train and Val must have same features"
    !isempty(testX) && @assert size(trainX,2) == size(testX,2)  "Train and Test must have same features"

    # --- Build neural network using function ---
    ann = buildClassANN(size(trainX,2), topology, size(trainY,2); transferFunctions=transferFunctions)

    # --- Setting up the loss function ---
    loss(model, x, y) = size(y,1) == 1 ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # --- Initialization ---
    trainLossHistory, valLossHistory, testLossHistory = Float64[], Float64[], Float64[]
    bestValidationLoss = Inf
    numEpochsWithoutImprovement = 0
    numEpoch = 0
    bestANN = deepcopy(ann)

    # Save initial losses (epoch 0)
    trainLoss, valLoss, testLoss = calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
    push!(trainLossHistory, trainLoss)
    push!(valLossHistory, valLoss)
    push!(testLossHistory, testLoss)

    # Define the optimizer
    opt_state = Flux.setup(Adam(learningRate), ann)

    # --- Training loop ---
    while (numEpoch < maxEpochs) && (trainLoss > minLoss) && (numEpochsWithoutImprovement < maxEpochsVal)
        Flux.train!(loss, ann, [(trainX', trainY')], opt_state)

        numEpoch += 1
        
        trainLoss, valLoss, testLoss = calculateLossValues(ann, trainingDataset, validationDataset, testDataset, loss)
        push!(trainLossHistory, trainLoss)
        push!(valLossHistory, valLoss)
        push!(testLossHistory, testLoss)

        # Save the model with the lowest validation loss
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

    # If validation exists -> return best ANN
    finalANN = isempty(valX) ? ann : bestANN

    return finalANN, trainLossHistory, valLossHistory, testLossHistory
end

# Variant of trainClassANN with targets as vectors
function trainClassANN(topology::AbstractArray{<:Int,1},  
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
                    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
        maxEpochsVal::Int=20, showText::Bool=false)

    # --- Convert vector targets to column matrices ---
    trainX, trainY = trainingDataset
    trainY = reshape(trainY, :, 1)
    trainingDataset = (trainX, trainY)

    valX, valY = validationDataset
    valY = reshape(valY, :, 1)
    validationDataset = (valX, valY)

    testX, testY = testDataset
    testY = reshape(testY, :, 1)
    testDataset = (testX, testY)

    # --- Call the original function ---
    return trainClassANN(
        topology,
        trainingDataset=trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal,
        showText=showText
    )
end


# ---------------------------------------------------------
# -------------- Unit 4: Confusion Matrices ---------------
# ---------------------------------------------------------

# 1. Binary vector inputs
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length"

    # Confusion matrix components
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)

    # Metrics
    total = TP + TN + FP + FN
    accuracy = total == 0 ? 0.0 : (TP + TN) / total
    error_rate = total == 0 ? 0.0 : (FP + FN) / total
    sensitivity = (TP + FN) == 0 ? (TN == total ? 1.0 : 0.0) : TP / (TP + FN)
    specificity = (TN + FP) == 0 ? (TP == total ? 1.0 : 0.0) : TN / (TN + FP)
    ppv = (TP + FP) == 0 ? (TN == total ? 1.0 : 0.0) : TP / (TP + FP)
    npv = (TN + FN) == 0 ? (TP == total ? 1.0 : 0.0) : TN / (TN + FN)
    fscore = (sensitivity + ppv) == 0 ? 0.0 : 2 * (sensitivity * ppv) / (sensitivity + ppv)

    confMatrix = [TN FP; FN TP]

    return accuracy, error_rate, sensitivity, specificity, ppv, npv, fscore, confMatrix
end

# 1.b Real vector inputs (Threshold)
function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    boolOutputs = outputs .>= threshold
    return confusionMatrix(boolOutputs, targets)
end

# Print for vector inputs
function printConfusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    accuracy, error_rate, sensitivity, specificity, ppv, npv, fscore, confMatrix = confusionMatrix(outputs, targets)
    println("Metrics:")
    println("Accuracy: ", round(accuracy, digits=4))
    println("Error rate: ", round(error_rate, digits=4))
    println("Sensitivity (Recall): ", round(sensitivity, digits=4))
    println("Specificity: ", round(specificity, digits=4))
    println("PPV (Precision): ", round(ppv, digits=4))
    println("NPV: ", round(npv, digits=4))
    println("F-score: ", round(fscore, digits=4))
    println("\nConfusion Matrix:")
    println("       Predicted")
    println("       | 0 | 1 |")
    println("    --------------")
    println("Real 0 | $(confMatrix[1,1]) | $(confMatrix[1,2]) |")
    println("     1 | $(confMatrix[2,1]) | $(confMatrix[2,2]) |")
    println("")
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    accuracy, error_rate, sensitivity, specificity, ppv, npv, fscore, confMatrix = confusionMatrix(outputs, targets, threshold=threshold)
    println("Accuracy: $accuracy")
    println("Error rate: $error_rate")
    println("Sensitivity: $sensitivity")
    println("Specificity: $specificity")
    println("PPV: $ppv")
    println("NPV: $npv")
    println("F-score: $fscore")
    println("\nConfusion matrix:")
    println("       Predicted")
    println("       | 0 | 1 |")
    println("    --------------")
    println("Real 0 | $(confMatrix[1,1]) | $(confMatrix[1,2]) |")
    println("     1 | $(confMatrix[2,1]) | $(confMatrix[2,2]) |")
end

# 2. Multi-class Matrix inputs
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=false)
    @assert size(outputs) == size(targets) "Outputs and targets must have the same dimensions"
    numClasses = size(outputs, 2)

    if numClasses == 1
        return confusionMatrix(vec(outputs), vec(targets))
    elseif numClasses == 2
        error("Two-column matrices are invalid because they represent a redundant binary case.")
    end

    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    precision = zeros(numClasses)
    npv = zeros(numClasses)
    f1 = zeros(numClasses)
    validClasses = falses(numClasses)

    for i in 1:numClasses
        if sum(targets[:, i]) > 0
            _, _, sens, spec, ppv, npv_i, fscore, _ = confusionMatrix(outputs[:, i], targets[:, i])
            sensitivity[i] = sens
            specificity[i] = spec
            precision[i] = ppv
            npv[i] = npv_i
            f1[i] = fscore
            validClasses[i] = true
        end
    end

    confMatrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:numClasses, j in 1:numClasses]
    counts = vec(sum(targets, dims=1))

    if weighted
        total = sum(counts)
        wmean(x) = sum(counts .* x) / total
        sensFinal, specFinal, precFinal, npvFinal, f1Final = 
            wmean.(Ref(sensitivity)), wmean.(Ref(specificity)), wmean.(Ref(precision)), wmean.(Ref(npv)), wmean.(Ref(f1))
    else
        valid_idx = findall(validClasses)
        n_valid = length(valid_idx)
        sensFinal = n_valid > 0 ? sum(sensitivity[valid_idx]) / n_valid : NaN
        specFinal = n_valid > 0 ? sum(specificity[valid_idx]) / n_valid : NaN
        precFinal = n_valid > 0 ? sum(precision[valid_idx]) / n_valid : NaN
        npvFinal  = n_valid > 0 ? sum(npv[valid_idx]) / n_valid : NaN
        f1Final   = n_valid > 0 ? sum(f1[valid_idx]) / n_valid : NaN
    end

    accuracy = sum(diag(confMatrix)) / sum(confMatrix)
    error_rate = 1 - accuracy

    return accuracy, error_rate, sensFinal, specFinal, precFinal, npvFinal, f1Final, confMatrix
end

# 2.b Matrix Real inputs (Threshold + Classification)
function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    classified = classifyOutputs(outputs; threshold=threshold)
    return confusionMatrix(classified, targets; weighted=weighted)
end

# 3. Generic Inputs (with classes)
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length"
    @assert all(x -> x in classes, outputs) "All outputs must be in classes"
    @assert all(x -> x in classes, targets) "All targets must be in classes"

    outputs_onehot = oneHotEncoding(outputs, classes)
    targets_onehot = oneHotEncoding(targets, classes)
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end

# 3.b Generic Inputs (auto classes)
function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    classes = unique(vcat(targets, outputs))
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end

# Print functions for Matrices/Generic
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    accuracy, error_rate, sensitivity, specificity, PPV, NPV, F1, conf_matrix = confusionMatrix(outputs, targets; weighted=weighted)
    println("Confusion Matrix:"); println(conf_matrix); println()
    println("Accuracy: ", accuracy); println("Error rate: ", error_rate)
    println("Sensitivity: ", sensitivity); println("Specificity: ", specificity)
    println("PPV: ", PPV); println("NPV: ", NPV); println("F1-score: ", F1)
end

function printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    accuracy, error_rate, sensitivity, specificity, PPV, NPV, F1, conf_matrix = confusionMatrix(outputs, targets; threshold=threshold, weighted=weighted)
    println("Confusion Matrix:"); println(conf_matrix); println()
    println("Accuracy: ", accuracy); println("Error rate: ", error_rate)
    println("Sensitivity: ", sensitivity); println("Specificity: ", specificity)
    println("PPV: ", PPV); println("NPV: ", NPV); println("F1-score: ", F1)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    accuracy, error_rate, sensitivity, specificity, PPV, NPV, F1, conf_matrix = confusionMatrix(outputs, targets, classes; weighted=weighted)
    println("Confusion Matrix:"); println(conf_matrix); println()
    println("Accuracy: ", accuracy); println("Error rate: ", error_rate)
    println("Sensitivity: ", sensitivity); println("Specificity: ", specificity)
    println("PPV: ", PPV); println("NPV: ", NPV); println("F1-score: ", F1)
end

function printConfusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    accuracy, error_rate, sensitivity, specificity, PPV, NPV, F1, conf_matrix = confusionMatrix(outputs, targets; weighted=weighted)
    println("Confusion Matrix:"); println(conf_matrix); println()
    println("Accuracy: ", accuracy); println("Error rate: ", error_rate)
    println("Sensitivity: ", sensitivity); println("Specificity: ", specificity)
    println("PPV: ", PPV); println("NPV: ", NPV); println("F1-score: ", F1)
end


# ---------------------------------------------------------
# -------------- Unit 5: Cross Validation -----------------
# ---------------------------------------------------------

function crossvalidation(N::Int64, k::Int64)
    folds = collect(1:k)
    indices = repeat(folds, ceil(Int, N / k))
    indices = indices[1:N]
    shuffle!(indices)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = Array{Int64,1}(undef, size(targets,1))
    pos_idx = findall(targets .== true)
    neg_idx = findall(targets .== false)
    indices[pos_idx] = crossvalidation(length(pos_idx), k)
    indices[neg_idx] = crossvalidation(length(neg_idx), k)
    return indices
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    indices = Array{Int64,1}(undef, size(targets,1))
    for i in 1:size(targets,2)
        n_class = sum(targets[:, i])
        @assert n_class >= k "Class $i has fewer than $k instances and it is not possible to perform $k-fold cross-validation."
        fold_assignments = crossvalidation(n_class, k)
        indices[targets[:, i] .== true] = fold_assignments
    end
    return indices
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    encoded = oneHotEncoding(targets)
    return crossvalidation(encoded, k)
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    
    # 0. Obtain inputs and targets
    inputs, targets = dataset # 'inputs' is UNNORMALIZED

    # 1. Extract class labels
    classes = unique(targets)
    numClasses = length(classes)

    # 2. One-hot encode the categorical target labels
    outputs = oneHotEncoding(targets, classes)

    # 3. Determine the number of folds
    numFolds = maximum(crossValidationIndices)

    # 4. Create one vector per metric
    accuracy_fold = Array{Float64,1}(undef, numFolds)
    error_fold = Array{Float64,1}(undef, numFolds)
    sensitivity_fold = Array{Float64,1}(undef, numFolds)
    specificity_fold = Array{Float64,1}(undef, numFolds)
    ppv_fold = Array{Float64,1}(undef, numFolds)
    npv_fold = Array{Float64,1}(undef, numFolds)
    f1_fold = Array{Float64,1}(undef, numFolds)

    # 5. Initialize a confusion matrix accumulator
    globalConfMatrix = zeros(Float64, numClasses, numClasses)

    ## For each fold
    for fold in 1:numFolds
        # 0. Extract 'train' and 'test' subsets
        train_idx = findall(x -> x != fold, crossValidationIndices)
        test_idx = findall(x -> x == fold, crossValidationIndices)

        X_train, Y_train = inputs[train_idx, :], outputs[train_idx, :]
        X_test, Y_test = inputs[test_idx, :], outputs[test_idx, :]

        # 1. Calculate normalization parameters using ONLY X_train (from this fold)
        normalizationParameters = calculateMinMaxNormalizationParameters(X_train)
        
        # 2. Apply those parameters to X_train and X_test (modifies in-place)
        normalizeMinMax!(X_train, normalizationParameters)
        normalizeMinMax!(X_test, normalizationParameters)

        # 1. Initialize vectors to store the metric results for each execution
        testAccuraciesEachRepetition = Array{Float64,1}(undef, numExecutions)
        testErrorsEachRepetition      = Array{Float64,1}(undef, numExecutions)
        testSensitivitiesEachRepetition = Array{Float64,1}(undef, numExecutions)
        testSpecificitiesEachRepetition = Array{Float64,1}(undef, numExecutions)
        testPPVsEachRepetition = Array{Float64,1}(undef, numExecutions)
        testNPVsEachRepetition = Array{Float64,1}(undef, numExecutions)
        testF1sEachRepetition  = Array{Float64,1}(undef, numExecutions)

        # 2. Create a 3D array for confusion matrices
        confMatrices = zeros(Float64, numClasses, numClasses, numExecutions)

        ## For each execution
        for numTraining in 1:numExecutions
            # Split training into training + validation if needed
            if validationRatio > 0
                train_sub_idx, val_idx = holdOut(size(X_train,1), validationRatio)
                
                # NOTE: X_train is already normalized, so X_train_sub and X_val are also normalized.
                X_train_sub, Y_train_sub = X_train[train_sub_idx, :], Y_train[train_sub_idx, :]
                X_val, Y_val = X_train[val_idx, :], Y_train[val_idx, :]
            else
                X_train_sub, Y_train_sub = X_train, Y_train
                X_val, Y_val = Array{Float64,2}(undef,0,0), falses(0, numClasses)
            end
            
            # Train ANN
            ann_model, _, _, _ = trainClassANN(
                topology,
                (X_train_sub, Y_train_sub),
                validationDataset=(X_val, Y_val),
                testDataset=(X_test, Y_test),
                transferFunctions=transferFunctions,
                maxEpochs=maxEpochs,
                minLoss=minLoss,
                learningRate=learningRate,
                maxEpochsVal=maxEpochsVal
            )

            # Obtain test outputs of the model
            outputs_test = ann_model(X_test') 
            
            # Evaluate test
            accuracy, error_rate, sens, spec, ppv, npv, f1, confMatrix = confusionMatrix(outputs_test', Y_test)

            # Store execution metrics
            testAccuraciesEachRepetition[numTraining] = accuracy
            testErrorsEachRepetition[numTraining] = error_rate
            testSensitivitiesEachRepetition[numTraining] = sens
            testSpecificitiesEachRepetition[numTraining] = spec
            testPPVsEachRepetition[numTraining] = ppv
            testNPVsEachRepetition[numTraining] = npv
            testF1sEachRepetition[numTraining] = f1
            confMatrices[:, :, numTraining] = confMatrix
        end
        
        # Average metrics per fold
        accuracy_fold[fold] = mean(testAccuraciesEachRepetition)
        error_fold[fold] = mean(testErrorsEachRepetition)
        sensitivity_fold[fold] = mean(testSensitivitiesEachRepetition)
        specificity_fold[fold] = mean(testSpecificitiesEachRepetition)
        ppv_fold[fold] = mean(testPPVsEachRepetition)
        npv_fold[fold] = mean(testNPVsEachRepetition)
        f1_fold[fold] = mean(testF1sEachRepetition)

        # Add mean confusion matrix to global
        meanConfMatrix = dropdims(mean(confMatrices, dims=3), dims=3)
        globalConfMatrix .+= meanConfMatrix
    end
    
    # Compute mean and std for each metric across folds
    results = (
        (mean(accuracy_fold), std(accuracy_fold)),
        (mean(error_fold), std(error_fold)),
        (mean(sensitivity_fold), std(sensitivity_fold)),
        (mean(specificity_fold), std(specificity_fold)),
        (mean(ppv_fold), std(ppv_fold)),
        (mean(npv_fold), std(npv_fold)),
        (mean(f1_fold), std(f1_fold)),
        globalConfMatrix
    )

    return results
end