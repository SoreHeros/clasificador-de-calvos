using Statistics
using Flux
using Flux.Losses
using Random

###### ONE HOT ENCODING #######

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    if numClasses == 0
        return falses(length(feature), 0)
    end
    
    if numClasses <= 2
        return reshape(feature .== classes[1], :, 1)
    else
        numPatterns = length(feature)
        oneHotResult = falses(numPatterns, numClasses)
        
        for i in 1:numClasses
            oneHotResult[:, i] .= (feature .== classes[i])
        end
        
        return oneHotResult
    end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

function oneHotEncoding(feature::AbstractArray{Bool,1})
    return reshape(feature, :, 1)
end

###### NORMALIZACION ####### 

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (minimum(dataset, dims=1), maximum(dataset, dims=1))
end

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return (mean(dataset, dims=1), std(dataset, dims=1))
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, 
                          normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues, maxValues = normalizationParameters
    dataset .-= minValues
    range = maxValues .- minValues
    range[range .== 0] .= 1
    dataset ./= range
    return nothing
end

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    params = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax!(dataset, params)
end

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
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs .>= threshold
end

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2)
    
    if numOutputs == 1
        return outputs .>= threshold
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        result = falses(size(outputs))
        result[indicesMaxEachInstance] .= true
        return result
    end
end

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean(outputs .== targets)
end

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(outputs, 2) == 1
        return accuracy(vec(outputs), vec(targets))
    else
        return mean(all(outputs .== targets, dims=2))
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

