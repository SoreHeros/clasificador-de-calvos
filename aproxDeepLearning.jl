using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, adjust!
using FileIO
using Statistics: mean
using Images, Random
include("Practicas/ejercicios/main.jl")

# Fijamos la semilla para reproducibilidad
Random.seed!(1)

function cargarDataset(directorioBase::String, clasesInteres::Vector{String})
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
            reduced = imresize(img, 28, 28)
            
            push!(entradas, Float32.(reduced))
            push!(etiquetas, clase)
            push!(rutas, rutaCompleta)
        end
    end
    return entradas, etiquetas, rutas
end


# Seleccionamos las clases
#clases_elegidas = ["nivel_$i" for i in 1:7]
clases_elegidas = ["nivel_1","nivel_3","nivel_5","nivel_7"]
entradas, etiquetas, rutas = cargarDataset("Dataset", clases_elegidas)

# Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz de 4 dimensiones donde esté almacenada la información de todas las imágenes
# Los formatos más habituales de estas matrices son NCHW y NHWC. Sin embargo, en Julia las imágenes se esperan en formato WHCN, como se puede ver en la documentación oficial en https://fluxml.ai/Flux.jl/stable/reference/models/layers/
#  Es decir, Width x Height x Channels x N
#  En el caso de esta base de datos
#   Height = 28
#   Width = 28
#   Channels = 1 -> son imagenes en escala de grises
#     Si fuesen en color, Channels = 3 (rojo, verde, azul)
# Esta conversion se puede hacer con la siguiente funcion:

function convertirArrayImagenesWHCN(imagenes)
    numPatrones = length(imagenes);
    nuevoArray = Array{Float32,4}(undef, 28, 28, 1, numPatrones); # Importante que sea un array de Float32
    for i in 1:numPatrones
        @assert (size(imagenes[i])==(28,28)) "Las imagenes no tienen tamaño 28x28";
        nuevoArray[:,:,1,i] .= imagenes[i][:,:]';
    end;
    return nuevoArray;
end;


println("Encontradas ", length(entradas), " imagenes")

# Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
#  Como primer elemento, las imagenes de ese batch
#     train_imgs[:,:,:,indicesBatch]
#  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
#     Para conseguir estas salidas deseadas, se hace una llamada a la funcion oneHotEncoding
#     oneHotEncoding(train_labels[indicesBatch], labels)'
#        El resultado se traspone, porque las instancias deben estar en las columnas, no en las filas
#  Por tanto, cada batch será un par dado por
#     (train_imgs[:,:,:,indicesBatch], oneHotEncoding(train_labels[indicesBatch], labels)')
# Sólo resta iterar por cada batch para construir el vector de batches
#train_set = [ (train_imgs[:,:,:,indicesBatch], oneHotEncoding(train_labels[indicesBatch], labels)') for indicesBatch in gruposIndicesBatch];

# Creamos un batch similar, pero con todas las imagenes de test
#test_set = (test_imgs, oneHotEncoding(test_labels, labels)');

funcionTransferenciaCapasConvolucionales = relu;

# Definimos la red con la funcion Chain, que concatena distintas capas
ann = Chain(
    # Primera capa: convolucion, que opera sobre una imagen 28x28
    # Argumentos:
    #  (3, 3): Tamaño del filtro de convolucion
    #  1=>16:
    #   1 canal de entrada: una imagen (matriz) de entradas
    #      En este caso, hay un canal de entrada porque es una imagen en escala de grises
    #      Si fuese, por ejemplo, una imagen en RGB, serian 3 canales de entrada
    #   16 canales de salida: se generan 16 filtros
    #  Es decir, se generan 16 imagenes a partir de la imagen original con filtros 3x3
    # Entradas a esta capa: matriz 4D de dimension 28 x 28 x 1canal    x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension 28 x 28 x 16canales x <numPatrones>
    Conv((3, 3), 1=>16, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    # Capa maxpool: es una funcion
    # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes 28x28 a 14x14
    # Entradas a esta capa: matriz 4D de dimension 28 x 28 x 16canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension 14 x 14 x 16canales x <numPatrones>
    MaxPool((2,2)),

    # Tercera capa: segunda convolucion: Le llegan 16 imagenes de tamaño 14x14
    #  16=>32:
    #   16 canales de entrada: 16 imagenes (matrices) de entradas
    #   32 canales de salida: se generan 32 filtros (cada uno toma entradas de 16 imagenes)
    #  Es decir, se generan 32 imagenes a partir de las 16 imagenes de entrada con filtros 3x3
    # Entradas a esta capa: matriz 4D de dimension 14 x 14 x 16canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension 14 x 14 x 32canales x <numPatrones>
    Conv((3, 3), 16=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    # Capa maxpool: es una funcion
    # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes 14x14 a 7x7
    # Entradas a esta capa: matriz 4D de dimension 14 x 14 x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension  7 x  7 x 32canales x <numPatrones>
    MaxPool((2,2)),

    # Tercera convolucion, le llegan 32 imagenes de tamaño 7x7
    #  32=>32:
    #   32 canales de entrada: 32 imagenes (matrices) de entradas
    #   32 canales de salida: se generan 32 filtros (cada uno toma entradas de 32 imagenes)
    #  Es decir, se generan 32 imagenes a partir de las 32 imagenes de entrada con filtros 3x3
    # Entradas a esta capa: matriz 4D de dimension 7 x 7 x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension 7 x 7 x 32canales x <numPatrones>
    Conv((3, 3), 32=>32, pad=(1,1), funcionTransferenciaCapasConvolucionales),

    # Capa maxpool: es una funcion
    # Divide el tamaño en 2 en el eje x y en el eje y: Pasa imagenes 7x7 a 3x3
    # Entradas a esta capa: matriz 4D de dimension 7 x 7 x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension 3 x 3 x 32canales x <numPatrones>
    MaxPool((2,2)),

    # Cambia el tamaño del tensot 3D en uno 2D
    #  Pasa matrices H x W x C x N a matrices H*W*C x N
    #  Es decir, cada patron de tamaño 3 x 3 x 32 lo convierte en un array de longitud 3*3*32
    # Entradas a esta capa: matriz 4D de dimension 3 x 3 x 32canales x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension 288 x <numPatrones>
    x -> reshape(x, :, size(x, 4)),

    # Capa totalmente conectada
    #  Como una capa oculta de un perceptron multicapa "clasico"
    #  Parametros: numero de entradas (288) y numero de salidas (10)
    #   Se toman 10 salidas porque tenemos 10 clases (numeros de 0 a 9)
    # Entradas a esta capa: matriz 4D de dimension 288 x <numPatrones>
    # Salidas de esta capa: matriz 4D de dimension  10 x <numPatrones>
    Dense(288, length(clases_elegidas)),

    # Finalmente, capa softmax
    #  Toma las salidas de la capa anterior y aplica la funcion softmax de tal manera
    #   que las 10 salidas sean valores entre 0 y 1 con las probabilidades de que un patron
    #   sea de una clase determinada (es decir, las probabilidades de que sea un digito determinado)
    #  Y, ademas, la suma de estas probabilidades sea igual a 1
    softmax
)

# Cuidado: En esta RNA se aplica la funcion softmax al final porque se tienen varias clases
# Si sólo se tuviesen 2 clases, solo se tiene una salida, y no seria necesario utilizar la funcion softmax
#  En su lugar, la capa totalmente conectada tendria como funcion de transferencia una sigmoidal (devuelve valores entre 0 y 1)
#  Es decir, no habria capa softmax, y la capa totalmente conectada seria la ultima, y seria Dense(288, 1, σ)
# @assert(size(train_set[1][2],1)>2, "RNA mal construida, para 2 clases")

indices = crossvalidation(length(entradas), 10)
labels = unique(etiquetas)

#folds
for numFold in 1:10

    #separar fold
    train_set = ((convertirArrayImagenesWHCN(entradas[indices.!=numFold,:,:,:]), oneHotEncoding(etiquetas[indices.!=numFold], labels)'))
    test_set = ((convertirArrayImagenesWHCN(entradas[indices.==numFold,:,:,:]), oneHotEncoding(etiquetas[indices.==numFold], labels)'))

    println(typeof(train_set))

    entradaCapa = train_set[1][:,:,:,[2, 5, 7]];
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

    # Sin embargo, para aplicar un patron no hace falta hacer todo eso.
    #  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
    ann(train_set[1][:,:,:,[2, 5, 7]]);



    # Valores de L1 y L2 para hacer regularización (probar distintos valores)
    L1 = 0;
    L2 = 0;

    # Definimos la funcion de loss de forma similar a la práctica 1 de la asignatura. Sin embargo, aquí os dejamos la posibilidad de usar regularización L1/L2
    absnorm(x) = sum(abs , x)
    sqrnorm(x) = sum(abs2, x)
    loss(ann, x, y) = ((size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y)) + L1*sum(absnorm, Flux.params(ann)) + L2*sum(sqrnorm, Flux.params(ann))
    # Para calcular la precisión, hacemos una llamada a la función accuracy con las salidas del la RNA y las salidas deseadas, de un batch concreto
    #  Se trasponen las matrices porque la función accuracy fue desarrollada esperando que las instancias estén en las filas
    #accuracy(batch) = accuracy(ann(batch[1])', batch[2]');
    # Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


    # Mostramos la precision antes de comenzar el entrenamiento:
    #  train_set es un array de batches
    #  accuracy recibe como parametro un batch
    #  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
    #   y devuelve un array con los resultados
    #  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
    #   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
    #println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy(ann(train_set[1])', train_set[2]')), " %");
    
    
    # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
    eta = 0.01;
    opt_state = Flux.setup(Adam(eta), ann);


    println("Comenzando entrenamiento...")
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
        precisionEntrenamiento = mean(accuracy.(train_set));
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento > mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            precisionTest = accuracy(test_set);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (eta > 1e-6)
            eta /= 10.0
            println("   No se ha mejorado la precision en el conjunto de entrenamiento en 5 ciclos, se baja la tasa de aprendizaje a ", eta);
            adjust!(opt_state, eta)
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end

    #acumular matriz de confusión y mostrarla fuera del bucle

end

# Finalmente, imprimimos la matriz de confusión en test junto con las distintas métricas
#  Para esto, se utilizan varias funciones desarrolladas en la práctica 1
#  Las matrices de salidas y salidas deseadas están traspuestas porque las funciones desarrolladas esperan instancias en filas, no en columnas
printConfusionMatrix(classifyOutputs(ann(test_set[1])'), test_set[2]');

return nothing;