# preprocessings.jl
# Cada función es del tipo:
#   (Xtr, Xte, ytr) -> (Xtr_prep, Xte_prep, state)

using .Utils
using MLJ
using Tables

# Cargamos el modelo PCA de MLJ una vez, a nivel global
const PCA_model = @load PCA pkg=MultivariateStats verbosity=0
const ICA_model = @load ICA pkg=MultivariateStats verbosity=0
const LDA_model = @load LDA pkg=MultivariateStats verbosity=0

"""
    preprocessing_1(Xtr, Xte, ytr)

MinMax sobre todas las features, usando parámetros calculados en Xtr.
"""
function preprocessing_1(
    Xtr::AbstractMatrix,
    Xte::AbstractMatrix,
    ytr
)
    Xtr2 = copy(Xtr)
    Xte2 = copy(Xte)

    params = Utils.calculateMinMaxNormalizationParameters(Xtr2)
    Utils.normalizeMinMax!(Xtr2, params)
    Utils.normalizeMinMax!(Xte2, params)

    state = (norm_params = params,)
    return Xtr2, Xte2, state
end

"""
    preprocessing_2(Xtr, Xte, ytr)

MinMax + PCA, manteniendo el 95% de la varianza por defecto.
"""
function preprocessing_2(
    Xtr::AbstractMatrix,
    Xte::AbstractMatrix,
    ytr;
    variance_ratio = 0.95
)
    # 1. MinMax
    Xtr2 = copy(Xtr)
    Xte2 = copy(Xte)

    params = Utils.calculateMinMaxNormalizationParameters(Xtr2)
    Utils.normalizeMinMax!(Xtr2, params)
    Utils.normalizeMinMax!(Xte2, params)

    # 2. PCA usando el modelo cargado globalmente
    pca_model = PCA_model(variance_ratio = variance_ratio)
    mach = machine(pca_model, Xtr2)   # Xtr2 matriz; MLJ la acepta como tabla
    fit!(mach, verbosity=0)

    pca_train_tbl = MLJ.transform(mach, Xtr2)
    pca_test_tbl  = MLJ.transform(mach, Xte2)

    Xtr_pca = Tables.matrix(pca_train_tbl)
    Xte_pca = Tables.matrix(pca_test_tbl)

    state = (norm_params = params, pca_mach = mach)
    return Xtr_pca, Xte_pca, state
end

"""
    preprocessing_ica(Xtr, Xte, ytr; outdim=8)

MinMax + ICA (Independent Component Analysis) with 8 independent components.
"""

function preprocessing_ica(
    Xtr::AbstractMatrix,
    Xte::AbstractMatrix,
    ytr;
    outdim::Int = 8
)
    # 1) MinMax
    Xtr2 = copy(Xtr)
    Xte2 = copy(Xte)

    params = Utils.calculateMinMaxNormalizationParameters(Xtr2)
    Utils.normalizeMinMax!(Xtr2, params)
    Utils.normalizeMinMax!(Xte2, params)

    # 2) ICA vía MLJ 
    ica_model = ICA_model(
        outdim = outdim,
        maxiter = 1000,    
        tol     = 1e-4)
    ica_mach  = machine(ica_model, Xtr2)
    fit!(ica_mach, verbosity = 0)

    ica_train_tbl = MLJ.transform(ica_mach, Xtr2)
    ica_test_tbl  = MLJ.transform(ica_mach, Xte2)

    Xtr_ica = Tables.matrix(ica_train_tbl)
    Xte_ica = Tables.matrix(ica_test_tbl)

    state = (norm_params = params, ica_mach = ica_mach)
    return Xtr_ica, Xte_ica, state
end

"""
    preprocessing_lda(Xtr, Xte, ytr)

MinMax + LDA (Linear Discriminant Analysis).
"""
function preprocessing_lda(
    Xtr::AbstractMatrix,
    Xte::AbstractMatrix,
    ytr
)
    # 1) MinMax
    Xtr2 = copy(Xtr)
    Xte2 = copy(Xte)

    params = Utils.calculateMinMaxNormalizationParameters(Xtr2)
    Utils.normalizeMinMax!(Xtr2, params)
    Utils.normalizeMinMax!(Xte2, params)

    # 2) LDA
    Xtr_f64 = convert(Matrix{Float64}, Xtr2)
    Xte_f64 = convert(Matrix{Float64}, Xte2)
    ytr_cat = categorical(ytr)

    lda_model = LDA_model()
    lda_mach  = machine(lda_model, MLJ.table(Xtr_f64), ytr_cat)
    fit!(lda_mach, verbosity = 0)

    lda_train_tbl = MLJ.transform(lda_mach, MLJ.table(Xtr_f64))
    lda_test_tbl  = MLJ.transform(lda_mach, MLJ.table(Xte_f64))

    Xtr_lda = Tables.matrix(lda_train_tbl)
    Xte_lda = Tables.matrix(lda_test_tbl)

    state = (norm_params = params, lda_mach = lda_mach)
    return Xtr_lda, Xte_lda, state
end


"""
    preprocessing_identity(Xtr, Xte, ytr)

No hace nada: deja los datos tal cual.
"""
function preprocessing_identity(
    Xtr::AbstractMatrix,
    Xte::AbstractMatrix,
    ytr
)
    return Xtr, Xte, nothing
end

# Opcional: diccionario auxiliar si te interesa
const ALL_PREPROCESSINGS = Dict(
    :preprocessing_1       => preprocessing_1,
    :preprocessing_2       => preprocessing_2,
    :preprocessing_identity => preprocessing_identity
)
