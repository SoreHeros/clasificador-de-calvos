using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using JLD2, FileIO
using Statistics: mean
using Images, Random
include("Practicas/ejercicios/main.jl")

# Fijamos la semilla para reproducibilidad
Random.seed!(1)

function cargarDatasetGray(directorioBase::String, clasesInteres::Vector{String}, size)
    entradas = Vector{Array{Float32, 2}}(undef, 0)
    etiquetas = String[]
    rutas = String[]
    for clase in clasesInteres
        rutaClase = joinpath(directorioBase, clase)
        !isdir(rutaClase) && continue
        for archivo in readdir(rutaClase)
            if !any(ext -> endswith(lowercase(archivo), ext), [".png", ".jpg", ".jpeg"])
                continue
            end
            rutaCompleta = joinpath(rutaClase, archivo)
            img = Gray.(FileIO.load(rutaCompleta))
            reduced = imresize(img, size, size)
            
            push!(entradas, Float32.(reduced))
            push!(etiquetas, clase)
            push!(rutas, rutaCompleta)
        end
    end
    return entradas, etiquetas, rutas
end

function cargarDatasetRGB(directorioBase::String, clasesInteres::Vector{String}, size)
    entradas = Vector{Array{Float32, 3}}(undef, 0)
    etiquetas = String[]
    rutas = String[]
    for clase in clasesInteres
        rutaClase = joinpath(directorioBase, clase)
        !isdir(rutaClase) && continue
        for archivo in readdir(rutaClase)
            if !any(ext -> endswith(lowercase(archivo), ext), [".png", ".jpg", ".jpeg"])
                continue
            end
            rutaCompleta = joinpath(rutaClase, archivo)
            img = FileIO.load(rutaCompleta)
            reduced = permutedims(channelview(imresize(img, size, size)),[2,3,1])
            
            push!(entradas, Float32.(reduced))
            push!(etiquetas, clase)
            push!(rutas, rutaCompleta)
        end
    end
    return entradas, etiquetas, rutas
end

function convertirArrayImagenesWHCNGray(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, size(imagenes[1])[1], size(imagenes[1])[1], 1, numPatrones); # n x n pixels, 1 color, numPatrones imágenes
    for i in 1:numPatrones
        nuevoArray[:,:,1,i] .= imagenes[i][:,:]';
    end;
    return nuevoArray;
end;

function convertirArrayImagenesWHCNRGB(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, size(imagenes[1])[1], size(imagenes[1])[1], 3, numPatrones); # n x n pixels, 3 colores, numPatrones imágenes
    for i in 1:numPatrones
        nuevoArray[:,:,1,i] .= imagenes[i][:,:,1]';
        nuevoArray[:,:,2,i] .= imagenes[i][:,:,2]';
        nuevoArray[:,:,3,i] .= imagenes[i][:,:,3]';
    end;
    return nuevoArray;
end;

function genericExecute(ann, dataset, transformationFunction, folds)
    entradas, etiquetas, rutas = dataset()

    println("Encontradas ", length(entradas), " imagenes")


    indices = crossvalidation(length(entradas), 10)
    labels = unique(etiquetas)
    data = nothing
    bestAccuracy = 0
    bestFold = 0

    #folds
    for numFold in 1:folds

        #separar fold
        train_set = (transformationFunction(entradas[indices.!=numFold,:,:,:]), oneHotEncoding(etiquetas[indices.!=numFold], labels)')
        test_set  = (transformationFunction(entradas), oneHotEncoding(etiquetas, labels)')

        if(numFold == 1)
            entradaCapa = train_set[1][:,:,:,[1]];
            numCapas = length(Flux.params(ann));
            println("La RNA tiene ", numCapas, " capas:");
            for numCapa in 1:numCapas
                println("   Capa ", numCapa, ": ", ann[numCapa]);
                # Le pasamos la entrada a esta capa
                capa = ann[numCapa];
                salidaCapa = capa(entradaCapa);
                println("      La salida de esta capa tiene dimension ", size(salidaCapa));
                entradaCapa = salidaCapa;
            end
            println()
            println("Comenzando entrenamiento (esto podría tardar varios minutos)...")
            println()
        end

        #Loss viejo porque L1/L2 da problemas
        loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

        accuracyAux(batch) = accuracy(ann(batch[1])', batch[2]');

        #println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*accuracyAux(train_set), " %");

        # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
        eta = 0.01;
        opt_state = Flux.setup(Adam(eta), ann);


        print("Fold ", numFold, ": ")

        mejorPrecision = -Inf;
        criterioFin = false;
        numCiclo = 0;
        numCicloUltimaMejora = 0;
        mejorModelo = nothing;

        while !criterioFin
            # Se entrena un ciclo
            Flux.train!(loss, ann, [train_set], opt_state);

            numCiclo += 1;

            # Se calcula la precision en el conjunto de entrenamiento:
            precisionEntrenamiento = accuracyAux(train_set);
            #println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

            # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
            if (precisionEntrenamiento > mejorPrecision)
                mejorPrecision = precisionEntrenamiento;
                precisionTest = accuracyAux(test_set);
                #println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
                mejorModelo = deepcopy(ann);
                numCicloUltimaMejora = numCiclo;
            end

            # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
            if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
                eta /= 10.0
                #println("   No se ha mejorado la precision en el conjunto de entrenamiento en 5 ciclos, se baja la tasa de aprendizaje a ", eta);
                adjust!(opt_state, eta)
                numCicloUltimaMejora = numCiclo;
            end

            # Criterios de parada:

            # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
            if (precisionEntrenamiento >= 0.999)
                #println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
                criterioFin = true;
            end

            # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
            if (numCiclo - numCicloUltimaMejora >= 10)
                #println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
                criterioFin = true;
            end
            
                    #guardar mejor data
            if(mejorPrecision > bestAccuracy)
                bestAccuracy = mejorPrecision
                data = confusionMatrix(ann(test_set[1])', test_set[2]')
                bestFold = numFold
            end

        end

        println("Ciclos: ", numCiclo, " Precision: ", 100*mejorPrecision, " %")
    end

    # Mostrar mejores datos

    println("\nMejor Fold(",bestFold,"):")
    println("Accuracy: ", data[1], "\tErrorRate: ", data[2], "\tRecall: ", data[3], "\tSpecificity: ", data[4], "\tPrecision: ", data[5], "\tNPV: ", data[6], "\tF1: ", data[7])

    println("\nMATRIZ DE CONFUSIÓN")
    println("Filas = clase real, Columnas = clase predicha\n")

    # Cabecera
    print(" " ^ 12)
    for c in labels
        print(rpad(c, 12))
    end
    println()
    println("-" ^ (12 + 12 * length(labels)))

    for (i, real) in enumerate(labels)
        print(rpad(real, 12))
        for j in 1:length(labels)
            print(rpad(round(Int, data[8][i,j]), 12))
        end
        println()
    end
    
    println()
    println()
    println()
    println()
end

# Seleccionar las clases
#clases_elegidas = ["nivel_1","nivel_2","nivel_3","nivel_4","nivel_5","nivel_6","nivel_7"]
clases_elegidas = ["nivel_1","nivel_3","nivel_5","nivel_7"]

funcionTransferenciaCapasConvolucionales = relu;



println("8 x 8 pixels grises")
ann8 = Chain(
    #Convolición inicial 8 x 8 pixels x 1 canal ---> 8 x 8 pixels x 16 canales
    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar las imagenes de 8 x 8 a 4 x 4
    MaxPool((2,2)),
    #segunda convolución y aumentar los canales de 16 a 32
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #serializar matriz 4 x 4 x 32 ---> 512
    x -> reshape(x, :, size(x, 4)),
    # Capa totalmente conectada 512 ---> 7 (o las salidas correspondientes)
    Dense(512, length(clases_elegidas)),
    # Finalmente, capa softmax
    softmax
)

dataset8() = cargarDatasetGray("Dataset", clases_elegidas, 8)

genericExecute(ann8, dataset8, convertirArrayImagenesWHCNGray, 3)




println("8 x 8 pixels RGB")
annRGB = Chain(
    #Convolición inicial 8 x 8 pixels x 3 canales ---> 8 x 8 pixels x 16 canales
    Conv((3, 3), 3=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar las imagenes de 8 x 8 a 4 x 4
    MaxPool((2,2)),
    #segunda convolución y aumentar los canales de 16 a 32
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #serializar matriz 4 x 4 x 32 ---> 512
    x -> reshape(x, :, size(x, 4)),
    # Capa totalmente conectada 512 ---> 7 (o las salidas correspondientes)
    Dense(512, length(clases_elegidas)),
    # Finalmente, capa softmax
    softmax
)

datasetRGB() = cargarDatasetRGB("Dataset", clases_elegidas, 8)

genericExecute(annRGB, datasetRGB, convertirArrayImagenesWHCNRGB, 3)





println("16 x 16 pixels grises")
ann16 = Chain(
    #Convolición inicial 16 x 16 pixels x 1 canal ---> 16 x 16 pixels x 16 canales
    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar las imagenes de 16 x 16 a 8 x 8
    MaxPool((2,2)),
    #segunda convolución y aumentar los canales de 16 a 32
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar de 8 x 8 a 4 x 4
    MaxPool((2,2)),
    #serializar matriz 4 x 4 x 32 ---> 512
    x -> reshape(x, :, size(x, 4)),
    # Capa totalmente conectada 512 ---> 7 (o las salidas correspondientes)
    Dense(512, length(clases_elegidas)),
    # Finalmente, capa softmax
    softmax
)

dataset16() = cargarDatasetGray("Dataset", clases_elegidas, 16)

genericExecute(ann16, dataset16, convertirArrayImagenesWHCNGray, 3)







println("32 x 32 pixels grises")
ann32 = Chain(
    #Convolición inicial 32 x 32 pixels x 1 canal ---> 32 x 32 pixels x 16 canales
    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar las imagenes de 32 x 32 a 16 x 16
    MaxPool((2,2)),
    #segunda convolución y aumentar los canales de 16 a 32
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar de 16 x 16 a 8 x 8
    MaxPool((2,2)),
    #tercera convolución
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),
    #menguar de 8 x 8 a 4 x 4
    MaxPool((2,2)),
    #serializar matriz 4 x 4 x 32 ---> 512
    x -> reshape(x, :, size(x, 4)),
    # Capa totalmente conectada 512 ---> 7 (o las salidas correspondientes)
    Dense(512, length(clases_elegidas)),
    # Finalmente, capa softmax
    softmax
)

dataset32() = cargarDatasetGray("Dataset", clases_elegidas, 32)

genericExecute(ann32, dataset32, convertirArrayImagenesWHCNGray, 3)

return nothing;