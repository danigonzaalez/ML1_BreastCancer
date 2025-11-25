#############################
#   Preprocessing Utils    #
#############################

# ========== One-hot encoding ==========

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


# ========== Normalización Min-Max ==========

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

# ========== Clasificación de salidas (threshold / argmax) ==========

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

# ========== Hold-out splits ==========

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