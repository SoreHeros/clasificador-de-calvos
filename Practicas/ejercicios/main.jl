using Statistics
using Flux
using Flux.Losses
using Random
using SymDoME

# EJERCICIO 2

###### ONE HOT ENCODING #######
function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)

    # no lo pide pero si no hay elementos, matriz vacia
    if numClasses == 0
        return falses(length(feature), 0)
    end

    # clasificacion binaria 0 1
    if numClasses <= 2
        return reshape(feature .== classes[1], :, 1)
    else
        #multiclase
        oneHotResult = falses(length(feature), numClasses)

        for i in 1:numClasses
            oneHotResult[:, i] .= (feature .== classes[i])
        end

        return oneHotResult
    end
end

# extraer las clases de un vector con unique(feature)
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

#convierte en matriz de 1 col con reshape
function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)
end

###### NORMALIZACION ####### 

# de cada col calcula el min y el max y lo mete en tupla
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims=1), maximum(dataset, dims=1))
end

# de cada col calcula la media y la desviacion y la emte en una tupla
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (mean(dataset, dims=1), std(dataset, dims=1))
end

# modificamos la matriz original, normalizamos datos entre 0 y 1 con la formula
function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    range = maxValues .- minValues
    range[range.==0] .= 1
    dataset ./= range
    return dataset
end

# version automatica de lo de arriba
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    params = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, params)
end

# version que no modifica la matriz, sino q devuevla una matriz nueva
function normalizeMinMax(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    newDataset = copy(dataset)
    normalizeMinMax!(newDataset, normalizationParameters)
    return newDataset
end

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    params = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax(dataset, params)
end

# mismo que minmax pero con la Z-score
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    meanValues, stdValues = normalizationParameters
    dataset .-= meanValues
    stdValues[stdValues.==0] .= 1
    dataset ./= stdValues
    return dataset
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    params = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, params)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2},
    normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    newDataset = copy(dataset)
    normalizeZeroMean!(newDataset, normalizationParameters)
    return newDataset
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    params = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, params)
end

###### CLASIFICACION #####
# convierte las probabilidades en valores binarios
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2)

    if numOutputs == 1
        return reshape(classifyOutputs(outputs[:]; threshold=threshold), :, 1)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        result = falses(size(outputs))
        result[indicesMaxEachInstance] .= true
        return result
    end
end

# calcula % de accuracy
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        return accuracy(vec(outputs), vec(targets))
    else
        return mean(eachrow(outputs) .== eachrow(targets))
    end
end

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};
    threshold::Real=0.5)
    return accuracy(classifyOutputs(outputs; threshold=threshold), targets)
end

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
    threshold::Real=0.5)
    return accuracy(classifyOutputs(outputs; threshold=threshold), targets)
end

#### BUILD ANN ####

function buildClassANN(numInputs::Int,
    topology::AbstractArray{<:Int,1},
    numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

    @assert length(transferFunctions) == length(topology)

    ann = Chain()
    numInputsLayer = numInputs

    for i in eachindex(topology)
        ann = Chain(ann..., Dense(numInputsLayer, topology[i], transferFunctions[i]))
        numInputsLayer = topology[i]
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity))
        ann = Chain(ann..., softmax)
    end

    return ann
end

### TRAIN ANN ###

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    # Semilla fija para reproducibilidad
    Random.seed!(1)

    # Historial de pérdida
    lossHistory = Float32[]

    # Extraer inputs y targets
    inputs, targets = dataset

    # Verificar dimensiones
    @assert size(inputs, 1) == size(targets, 1) "Número de patrones inconsistente"

    # Construir la ANN
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)

    # Definir función de pérdida en la forma que sugiere el enunciado 
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # Preparar datos para Flux (transponer para que las columnas sean patrones)
    X = Float32.(inputs')
    Y = Float32.(targets')

    # Definir optimizador con Adam como sugiere el enunciado 
    opt_state = Flux.setup(Adam(learningRate), ann)

    # CICLO 0: Cálculo del error antes de empezar el entrenamiento 
    # Esto garantiza que el vector resultante tenga n+1 elementos 
    currentLoss = Float32(loss(ann, X, Y))
    push!(lossHistory, currentLoss)

    # Entrenamiento
    for epoch in 1:maxEpochs

        if currentLoss <= minLoss
            break
        end

        # Usamos Flux.train! como sugiere en el enunciado 
        Flux.train!(loss, ann, [(X, Y)], opt_state)

        # Calcular y almacenar pérdida
        currentLoss = Float32(loss(ann, X, Y))
        push!(lossHistory, currentLoss)

        if epoch % 100 == 0
            println("Época $epoch, Pérdida: $currentLoss")
        end
    end

    # Devolver la red entrenada y el historial de pérdida
    return (ann, lossHistory)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    # Convertir targets a matriz 2D (una columna)
    targetsMatrix = reshape(targets, :, 1)

    # Llamar a la versión anterior con formato correcto
    return trainClassANN(topology,
        (inputs, targetsMatrix);
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate)
end;

# EJERCICIO 3

function holdOut(N::Int, P::Real) #N numero de patrones y P porcentaje de patrones para el cojunto test
    index = randperm(N) #Generar un vector con numeros del 1 a N ordenados aleatoriamente 
    test_number = round(Int, N * (1 - P))
    return (index[1:test_number], index[test_number+1:N])
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    rest, v_test = holdOut(N, Ptest)
    new_pval = Pval * N / length(rest)
    v_train, v_val = holdOut(length(rest), new_pval)
    return (v_train, v_val, v_test)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0, size(trainingDataset[2], 2))),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0, size(trainingDataset[2], 2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    # Semilla fija para reproducibilidad
    Random.seed!(1)

    # Historial de pérdida
    lossHistory = Float32[]

    # Extraer y preparar datos de entrenamiento
    inputsTrain, targetsTrain = trainingDataset
    X_train = Float32.(inputsTrain')
    Y_train = Float32.(targetsTrain')

    # Verificar dimensiones
    @assert size(inputsTrain, 1) == size(targetsTrain, 1) "Número de patrones inconsistente"

    # Construir la ANN
    numInputs = size(inputsTrain, 2)
    numOutputs = size(targetsTrain, 2)
    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)

    # Definir función de pérdida en la forma que sugiere el enunciado 
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # Preparamos los datos de validación y test
    if size(validationDataset[1], 1) > 0
        X_val = Float32.(validationDataset[1]')
        Y_val = Float32.(validationDataset[2]')
    else
        X_val, Y_val = nothing, nothing
    end

    if size(testDataset[1], 1) > 0
        X_test = Float32.(testDataset[1]')
        Y_test = Float32.(testDataset[2]')
    else
        X_test, Y_test = nothing, nothing
    end

    # Definir optimizador con Adam como sugiere el enunciado 
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Inicializamos los 3 historiales
    trainLosses = Float32[]
    valLosses = Float32[]
    testLosses = Float32[]

    # Calculamos y guardamos el Ciclo 0 una sola vez por conjunto
    currentTrainLoss = Float32(loss(ann, X_train, Y_train))
    push!(trainLosses, currentTrainLoss)

    if !isnothing(X_val)
        currentValLoss = Float32(loss(ann, X_val, Y_val))
        push!(valLosses, currentValLoss)
        bestValLoss = currentValLoss # El mejor hasta ahora es el inicial
    else
        bestValLoss = Inf32 # Infinito si no hay validación para que no interfiera
    end

    if !isnothing(X_test)
        push!(testLosses, Float32(loss(ann, X_test, Y_test)))
    end

    # Inicializar mejor red
    bestAnn = deepcopy(ann)
    # Si hay validación, el mejor loss inicial es el de validación; si no, Inf porque siempre va a ser mayor que el loss de entrenamiento
    bestValLoss = !isnothing(X_val) ? Float32(loss(ann, X_val, Y_val)) : Inf32
    epochsWithoutImprovement = 0

    for epoch in 1:maxEpochs
        # Parada por error mínimo o parada temprana
        if currentTrainLoss <= minLoss || epochsWithoutImprovement >= maxEpochsVal
            break
        end

        # Entrenamiento
        Flux.train!(loss, ann, [(X_train, Y_train)], opt_state)

        # Actualizar historiales
        currentTrainLoss = Float32(loss(ann, X_train, Y_train))
        push!(trainLosses, currentTrainLoss)

        if !isnothing(X_val)
            currentValLoss = Float32(loss(ann, X_val, Y_val))
            push!(valLosses, currentValLoss)

            # Lógica de Early Stopping
            if currentValLoss < bestValLoss
                bestValLoss = currentValLoss
                bestAnn = deepcopy(ann)
                epochsWithoutImprovement = 0
            else
                epochsWithoutImprovement += 1
            end
        end

        if !isnothing(X_test)
            push!(testLosses, Float32(loss(ann, X_test, Y_test)))
        end
    end

    # Si hubo validación, devolvemos la mejor; si no, la última
    redADevolver = isnothing(X_val) ? ann : bestAnn
    return (redADevolver, trainLosses, valLosses, testLosses)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef, 0, size(trainingDataset[1], 2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)

    # Aplicamos reshape a los targets de los 3 conjuntos para convertirlos en matrices (N, 1)
    trainingDatasetMat = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    validationDatasetMat = (validationDataset[1], reshape(validationDataset[2], :, 1))
    testDatasetMat = (testDataset[1], reshape(testDataset[2], :, 1))

    # Llamamos a la función principal pasándole los nuevos datasets en formato matriz
    return trainClassANN(topology, trainingDatasetMat;
        validationDataset=validationDatasetMat,
        testDataset=testDatasetMat,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal)
end

# EJERCICIO 4

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Los vectores deben tener la misma longitud"

    TP = sum(outputs .& targets)
    TN = sum((.!outputs) .& (.!targets))
    FP = sum(outputs .& (.!targets))
    FN = sum((.!outputs) .& targets)

    total = TP + TN + FP + FN

    # Precisión (accuracy)
    acc = (TP + TN) / total

    # Tasa de error
    errorRate = (FP + FN) / total

    # Sensibilidad (recall) - manejar caso especial
    if TP + FN == 0
        recall = 1.0
    else
        recall = TP / (TP + FN)
    end

    # Especificidad (specificity) - manejar caso especial
    if TN + FP == 0
        specificity = 1.0
    else
        specificity = TN / (TN + FP)
    end

    # Valor predictivo positivo (precision) - manejar caso especial
    if TP + FP == 0
        precision = 1.0
    else
        precision = TP / (TP + FP)
    end

    # Valor predictivo negativo (NPV) - manejar caso especial
    if TN + FN == 0
        NPV = 1.0
    else
        NPV = TN / (TN + FN)
    end

    # F1-score - manejar caso especial
    if precision + recall == 0
        F1 = 0.0
    else
        F1 = 2 * (precision * recall) / (precision + recall)
    end

    # Matriz de confusión según el PDF: [TN FP; FN TP]
    # Es decir:
    # - Fila 1: Negativos reales (TN, FP)
    # - Fila 2: Positivos reales (FN, TP)
    confMatrix = [TN FP; FN TP]

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    return confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets)
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert size(outputs) == size(targets) "Las matrices deben tener las mismas dimensiones"
    @assert size(outputs, 2) == size(targets, 2) "El número de clases debe ser el mismo"

    (_, nClasses) = size(targets);

    if nClasses == 1
        # Caso binario: convertir a vectores y llamar a la versión 1D
        return confusionMatrix(vec(outputs), vec(targets))
    end

    # Calcular matriz de confusión: filas = clase real, columnas = clase predicha
    confMatrix_counts = targets' * outputs

    # Vectores para almacenar métricas por clase
    recall_vec = zeros(nClasses)
    specificity_vec = zeros(nClasses)
    precision_vec = zeros(nClasses)
    NPV_vec = zeros(nClasses)
    F1_vec = zeros(nClasses)

    # Métricas por clase
    for k in 1:nClasses
        (_, _, recall_vec[k], specificity_vec[k], precision_vec[k], NPV_vec[k], F1_vec[k], _) = confusionMatrix(outputs[:,k], targets[:,k]);
    end

    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;
    if weighted
        class_weights = vec(sum(targets, dims=1))
        total_instances = sum(class_weights)
        weights = class_weights./total_instances;

        recall = sum(weights .* recall_vec) 
        specificity = sum(weights .* specificity_vec)
        precision = sum(weights .* precision_vec)
        NPV = sum(weights .* NPV_vec) 
        F1 = sum(weights .* F1_vec) 

        # Devolver la matriz de conteos (enteros)
    else
        # Media aritmética (macro)
        recall = mean(recall_vec)
        specificity = mean(specificity_vec)
        precision = mean(precision_vec)
        NPV = mean(NPV_vec)
        F1 = mean(F1_vec)

    end
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix_counts)

end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    return confusionMatrix(classifyOutputs(outputs; threshold=threshold), targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # 1. Línea de programación defensiva obligatoria 
    @assert(all([in(label, classes) for label in vcat(targets, outputs)]))

    # 2. Transformamos a matrices one-hot SIN bucles como pide el enunciado 
    outputs_onehot = oneHotEncoding(outputs, classes)
    targets_onehot = oneHotEncoding(targets, classes)

    # 3. Llamamos a la versión de matrices booleanas 
    return confusionMatrix(outputs_onehot, targets_onehot; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Calculamos las clases únicas
    classes = unique(vcat(targets, outputs))
    # Llamamos a la versión anterior
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    entradas = Float64.(trainingDataset[1])
    targets = Vector{Bool}(trainingDataset[2])
    testEntradas = Float64.(testInputs)

    Random.seed!(1)
    (modelo, _, _, _) = dome(entradas, targets; maximumNodes=maximumNodes)

    testOutputs = evaluateTree(modelo, testEntradas)

    if isa(testOutputs, Real)
        testOutputs = repeat([testOutputs], size(testEntradas, 1))
    end

    return testOutputs
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    Random.seed!(1)
    entradas = Float64.(trainingDataset[1])
    targets = trainingDataset[2]
    testEntradas = Float64.(testInputs)
    numClasses = size(targets, 2)

    if (numClasses == 1)
        # Llamamos a la versión binaria (vector)
        resultado_vector = trainClassDoME((entradas, vec(targets)), testEntradas, maximumNodes)
        # Lo transformamos en matriz (N x 1) antes de devolverlo
        return reshape(resultado_vector, :, 1)
    else
        # 1. Crear matriz de resultados
        salidasTestMatriz = zeros(Float64, size(testEntradas, 1), numClasses)
        # 2. Bucle para cada clase 
        for i in 1:numClasses
            # 3. Llamada a la versión binaria con la columna i como vector 
            # 4. Asignación a la columna correspondiente 
            salidasTestMatriz[:, i] .= trainClassDoME((entradas, targets[:, i]), testEntradas, maximumNodes)
        end
        return salidasTestMatriz
    end
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    classes = unique(trainingDataset[2])
    # Creamos el vector con el mismo tipo de datos
    testOutputs = Array{eltype(trainingDataset[2]),1}(undef, size(testInputs, 1))

    # Obtenemos la matriz de certidumbres llamando a nuestra versión anterior usando oneHotEncoding
    testOutputsDOME = trainClassDoME(
        (trainingDataset[1], oneHotEncoding(trainingDataset[2], classes)),
        testInputs,
        maximumNodes
    )

    testOutputsBool = classifyOutputs(testOutputsDOME; threshold=0)

    if length(classes) <= 2
        # Caso 1-2 clases: Sin bucles
        # Primero convertimos la matriz de una columna en un vector para facilitar el mapeo
        testOutputsBool = vec(testOutputsBool)

        testOutputs[testOutputsBool] .= classes[1]

        if length(classes) == 2
            testOutputs[.!testOutputsBool] .= classes[2]
        end
    else
        # Caso > 2 clases: El bucle "Uno contra todos"
        for numClass in 1:length(classes)
            testOutputs[testOutputsBool[:, numClass]] .= classes[numClass]
        end
    end

    return testOutputs
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)


end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)


end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1},
    classes::AbstractArray{<:Any,1}; weighted::Bool=true)


end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)


end;

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random: seed!

function crossvalidation(N::Int64, k::Int64)
    v = collect(1:k)
    v = repeat(v, ceil(Int, N / k))
    v = v[1:N]
    shuffle!(v)
    return v
end

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    N = length(targets)
    @assert N >= k "Las clases deben tener minimo k patrones"
    v = collect(1:N)
    positive_in_class = sum(targets)
    negative_in_class = N - positive_in_class
    @assert (positive_in_class >= k && negative_in_class >= k) "Las clases deben tener minimo k patrones"
    v[targets] .= crossvalidation(positive_in_class, k)
    v[.!targets] .= crossvalidation(negative_in_class, k)
    return v
end

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    if size(targets, 2) == 1
        return crossvalidation(vec(targets), k)
    end
    N, num_classes = size(targets)
    v = collect(1:N)
    for class in 1:num_classes
        boolean_vector = targets[:, class]
        positive_in_class = sum(boolean_vector)
        @assert positive_in_class >= k "Las clases deben tener minimo k patrones"
        v[boolean_vector] .= crossvalidation(positive_in_class, k)
    end
    return v
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets)
    targets_onehot = oneHotEncoding(targets, classes)
    return crossvalidation(targets_onehot, k)
end

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)

    #Comprobar entradas
    @assert size(dataset[1], 1) == length(crossValidationIndices) "Debe de haber un índice por cada elemento de dataset"
    @assert size(dataset[1], 1) == length(dataset[2]) "Debe de haber una salida por cada entrada"
    #Get clases
    clases = unique(dataset[2])
    clasesLen = length(clases)
    if(clasesLen <= 2)
        clasesLen = 1
    end    
    #Get encoding
    encoding = oneHotEncoding(dataset[2], clases)
    #Get folds
    folds = maximum(crossValidationIndices)

    #Inits
    precision = zeros(folds)
    tasaError = zeros(folds)
    sensibilidad = zeros(folds)
    especificidad = zeros(folds)
    VPP = zeros(folds)
    VPN = zeros(folds)
    F1 = zeros(folds)
    confusionMatrixGlobal = zeros(Float32, length(clases), length(clases))
    for i in 1:folds
            #Separar entrenamiento (y validación) de test
            entradaEntrenamiento = eachrow(dataset[1])[crossValidationIndices .!= i]
            salidaDeseadaEntrenamiento = eachrow(encoding)[crossValidationIndices .!= i]
            entradaTest = eachrow(dataset[1])[crossValidationIndices .== i]
            salidaDeseadaTest = collect(eachrow(encoding)[crossValidationIndices .== i])
            
            #Inits
            precisionCarry = zeros(numExecutions)
            tasaErrorCarry = zeros(numExecutions)
            sensibilidadCarry = zeros(numExecutions)
            especificidadCarry = zeros(numExecutions)
            VPPCarry = zeros(numExecutions)
            VPNCarry = zeros(numExecutions)
            F1Carry = zeros(numExecutions)
            confusionMatrixCarry = Array{Array, 1}(undef, numExecutions)
            for j in 1:numExecutions

                #Separar entrenamiento y validación
                entrenamientoIndices, validationIndices = holdOut(length(entradaEntrenamiento), validationRatio * length(dataset[2]) / length(entradaEntrenamiento)) #validationRatio ajustado para que sea sobre el total
                entradaValidation = entradaEntrenamiento[validationIndices]
                salidaDeseadaValidation = salidaDeseadaEntrenamiento[validationIndices]
                entradaEntrenamientoMixed = entradaEntrenamiento[entrenamientoIndices]
                salidaDeseadaEntrenamientoMixed = collect(salidaDeseadaEntrenamiento[entrenamientoIndices])
                entrenamiento = (stack(entradaEntrenamientoMixed)', collect(stack(salidaDeseadaEntrenamientoMixed))')
                if(length(entradaValidation) == 0)
                    validation = nothing
                else
                    validation = (stack(entradaValidation)', collect(stack(salidaDeseadaValidation))')
                end
                test = (stack(entradaTest)', collect(stack(salidaDeseadaTest))')
                #Entrenar RNA
                if(isnothing(validation))
                    (redNeuronal, trainLosses, valLosses, testLosses) = trainClassANN(topology, entrenamiento, testDataset = test, transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate, maxEpochsVal = maxEpochsVal)
                else
                    (redNeuronal, trainLosses, valLosses, testLosses) = trainClassANN(topology, entrenamiento, validationDataset = validation, testDataset = test, transferFunctions = transferFunctions, maxEpochs = maxEpochs, minLoss = minLoss, learningRate = learningRate, maxEpochsVal = maxEpochsVal)
                end
                #Cojer salida de RNA
                salidaTest = redNeuronal(Float32.(stack(entradaTest)))
                #Salida y separación de confusionMatrix
                (precisionCarry[j], tasaErrorCarry[j], sensibilidadCarry[j], especificidadCarry[j], VPPCarry[j], VPNCarry[j], F1Carry[j], confusionMatrixCarry[j]) = confusionMatrix(stack(salidaTest)', collect(stack(salidaDeseadaTest))')
            end    

            #media hacha de forma manual
            precision[i] = mean(precisionCarry)
            tasaError[i] = mean(tasaErrorCarry)
            sensibilidad[i] = mean(sensibilidadCarry)
            especificidad[i] = mean(especificidadCarry)
            VPP[i] = mean(VPPCarry)
            VPN[i] = mean(VPNCarry)
            F1[i] = mean(F1Carry)
            confusionMatrixGlobal += mean(confusionMatrixCarry)
    end

    #media de valors con desviación + matriz de confusion total
    calc(vec) = (mean(vec), std(vec))
    return (calc(precision), calc(tasaError), calc(sensibilidad), calc(especificidad), calc(VPP), calc(VPN), calc(F1), confusionMatrixGlobal)
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg = LIBSVM verbosity = 0
kNNClassifier = MLJ.@load KNNClassifier pkg = NearestNeighborModels verbosity = 0
DTClassifier = MLJ.@load DecisionTreeClassifier pkg = DecisionTree verbosity = 0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    (inputs, targets) = dataset;

    # ANN
    if (modelType==:ANN)
        return ANNCrossValidation(modelHyperparameters["topology"],
        dataset,
        crossValidationIndices,
        numExecutions = haskey(modelHyperparameters, "numExecutions") ? modelHyperparameters["numExecutions"] : 50,
        transferFunctions = haskey(modelHyperparameters, "transferFunctions") ? modelHyperparameters["transferFunctions"] : fill(σ, length(modelHyperparameters["topology"])),
        maxEpochs = haskey(modelHyperparameters, "maxEpochs") ? modelHyperparameters["maxEpochs"] : 1000, 
        minLoss = haskey(modelHyperparameters, "minLoss") ? modelHyperparameters["minLoss"] : 0.0,
        learningRate = haskey(modelHyperparameters, "learningRate") ? modelHyperparameters["learningRate"] : 0.01,
        validationRatio = haskey(modelHyperparameters, "validationRatio") ? modelHyperparameters["validationRatio"] : 0,
        maxEpochsVal = haskey(modelHyperparameters, "maxEpochsVal") ? modelHyperparameters["maxEpochsVal"] : 20
        );
    end;

    # no ANN
    #salidas deseadas  -> vector de cadenas de texto
    targets = string.(targets);
    classes = unique(targets);

    #Get folds
    folds = maximum(crossValidationIndices)
    
    #Inits
    precision = Array{Float64,1}(undef, folds);
    tasaError = Array{Float64,1}(undef, folds);
    sensibilidad = Array{Float64,1}(undef, folds);
    especificidad = Array{Float64,1}(undef, folds);
    VPP = Array{Float64,1}(undef, folds);
    VPN = Array{Float64,1}(undef, folds);
    F1 = Array{Float64,1}(undef, folds);
    confusionMatrixGlobal = zeros(Float64, length(classes), length(classes), folds)
    

    for numFold in 1:folds

        #datos de entrenamiento y test
        trainingInputs = inputs[crossValidationIndices.!=numFold,:];
        testInputs = inputs[crossValidationIndices.==numFold,:];
        trainingTargets = targets[crossValidationIndices.!=numFold];
        testTargets = targets[crossValidationIndices.==numFold];

        if modelType==:DoME
            testOutputs = trainClassDoME((trainingInputs, trainingTargets), testInputs, modelHyperparameters["maximumNodes"]) 
        else
            if modelType==:SVC
                    @assert((modelHyperparameters["kernel"] == "linear") || (modelHyperparameters["kernel"] == "poly") || (modelHyperparameters["kernel"] == "rbf") || (modelHyperparameters["kernel"] == "sigmoid"));
                    model = SVMClassifier(
                        kernel = 
                            modelHyperparameters["kernel"]=="linear" ? LIBSVM.Kernel.Linear :
                            modelHyperparameters["kernel"]=="rbf" ? LIBSVM.Kernel.RadialBasis : 
                            modelHyperparameters["kernel"]=="poly" ? LIBSVM.Kernel.Polynomial :
                            modelHyperparameters["kernel"]=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing, 
                        cost   = Float64(modelHyperparameters["C"]),
                        gamma  = Float64(get(modelHyperparameters, "gamma",  -1)),
                        degree = Int32(  get(modelHyperparameters, "degree", -1)),
                        coef0  = Float64(get(modelHyperparameters, "coef0",  -1)));
            elseif modelType==:DecisionTreeClassifier
                model = DTClassifier(max_depth = modelHyperparameters["max_depth"], rng=Random.MersenneTwister(1));
            elseif modelType==:KNeighborsClassifier
                model = kNNClassifier(K = modelHyperparameters["n_neighbors"]);
            else
                error(string("Not valid model ", modelType));
            end;

            # entrenarmiento del modelo
            mach = machine(model, MLJ.table(trainingInputs), categorical(trainingTargets));
            MLJ.fit!(mach, verbosity=0)

            testOutputs = MLJ.predict(mach, MLJ.table(testInputs))

            if modelType==:DecisionTreeClassifier || modelType==:KNeighborsClassifier
                testOutputs = mode.(testOutputs)
            end;

        end;

        # Mertricas y matriz de confusiónm
        (precision[numFold],
        tasaError[numFold],
        sensibilidad[numFold],
        especificidad[numFold],
        VPP[numFold],
        VPN[numFold],
        F1[numFold],
        confusionMatrixGlobalfold) = confusionMatrix(testOutputs, testTargets, classes);
        confusionMatrixGlobal +=confusionMatrixGlobalfold
    end;

    #media de valors con desviación + matriz de confusion total
    calc(vec) = (mean(vec), std(vec))
    return (calc(precision), calc(tasaError), calc(sensibilidad), calc(especificidad), calc(VPP), calc(VPN), calc(F1), confusionMatrixGlobal)

end;