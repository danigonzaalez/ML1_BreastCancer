# preprocessings.jl
# Cada función es del tipo:
#   (Xtr, Xte, ytr) -> (Xtr_prep, Xte_prep, state)

using .Utils
using MLJ
using Tables

# Cargamos el modelo PCA de MLJ una vez, a nivel global
const PCA_model = @load PCA pkg=MultivariateStats verbosity=0

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
