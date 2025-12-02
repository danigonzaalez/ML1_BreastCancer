#############################
#       Metrics Utils      #
#############################

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets)
    TP = sum(outputs .& targets)
    TN = sum(.!outputs .& .!targets)
    FP = sum(outputs .& .!targets)
    FN = sum(.!outputs .& targets)
    beta = 2.0
    beta2 = beta^2
    total = TP + TN + FP + FN
    
    accuracy = total == 0 ? 0.0 : (TP + TN) / total
    error_rate = total == 0 ? 0.0 : (FP + FN) / total
    sensitivity = (TP + FN) == 0 ? (TN == total ? 1.0 : 0.0) : TP / (TP + FN)
    specificity = (TN + FP) == 0 ? (TP == total ? 1.0 : 0.0) : TN / (TN + FP)
    ppv = (TP + FP) == 0 ? (TN == total ? 1.0 : 0.0) : TP / (TP + FP)
    npv = (TN + FN) == 0 ? (TP == total ? 1.0 : 0.0) : TN / (TN + FN)
    fscore = (sensitivity + ppv) == 0 ? 0.0 : 2 * (sensitivity * ppv) / (sensitivity + ppv)
    f2score = (sensitivity + ppv) == 0 ? 0.0 :
              (1 + beta2) * sensitivity * ppv / (beta2 * ppv + sensitivity)
    
    return accuracy, error_rate, sensitivity, specificity, ppv, npv, fscore, f2score, [TN FP; FN TP]
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return confusionMatrix(outputs .>= threshold, targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=false)
    numClasses = size(outputs, 2)
    if numClasses == 1; return confusionMatrix(vec(outputs), vec(targets)); end
    if numClasses == 2; error("Two-column matrices are invalid (redundant binary case)."); end

    metrics = zeros(6, numClasses) # sens, spec, ppv, npv, f1
    validClasses = falses(numClasses)

    for i in 1:numClasses
        if sum(targets[:, i]) > 0
            _, _, metrics[1,i], metrics[2,i], metrics[3,i], metrics[4,i], metrics[5,i], metrics[6,i] = confusionMatrix(outputs[:, i], targets[:, i])
            validClasses[i] = true
        end
    end

    confMatrix = [sum(targets[:, i] .& outputs[:, j]) for i in 1:numClasses, j in 1:numClasses]
    
    if weighted
        counts = vec(sum(targets, dims=1)); total = sum(counts)
        finalMetrics = [sum(counts .* metrics[r,:]) / total for r in 1:6]
    else
        valid_idx = findall(validClasses); n_valid = length(valid_idx)
        finalMetrics = n_valid > 0 ? [sum(metrics[r,valid_idx]) / n_valid for r in 1:6] : fill(NaN, 6)
    end
    
    accuracy = sum(diag(confMatrix)) / sum(confMatrix)
    return accuracy, 1-accuracy, finalMetrics[1], finalMetrics[2], finalMetrics[3], finalMetrics[4], finalMetrics[5], finalMetrics[6], confMatrix
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
    println("F2-score: $(results[8])")
end