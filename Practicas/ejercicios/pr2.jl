using Statistics
using Flux
using Flux.Losses
using Random

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
                          normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    range = maxValues .- minValues
    range[range .== 0] .= 1
    dataset ./= range
    return nothing
end

# version automatica de lo de arriba
function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    params = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, params)
end

# version que no modifica la matriz, sino q devuevla una matriz nueva
function normalizeMinMax(dataset::AbstractArray{<:Real,2}, 
                         normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
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
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    meanValues, stdValues = normalizationParameters
    dataset .-= meanValues
    stdValues[stdValues .== 0] .= 1
    dataset ./= stdValues
    return nothing
end

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    params = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, params)
end

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, 
                           normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
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
                       dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
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
                       (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
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

