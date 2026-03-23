include("solucion.jl")
using Images, FileIO, Statistics, Random, Printf

# Fijamos la semilla 
Random.seed!(1);

function cargarDatasetCalvicie(rutaBase::String, tamanofijo::Tuple{Int,Int})
    entradas = []
    etiquetas = String[]

    # Obtenemos las carpetas de las clases
    clases = readdir(rutaBase)

    for clase in clases
        rutaClase = joinpath(rutaBase, clase)
        if isdir(rutaClase)
            for archivo in readdir(rutaClase)
                # Solo procesar imágenes
                if !any(ext -> endswith(lowercase(archivo), ext), [".png", ".jpg", ".jpeg"])
                    continue
                end

                # Cargar imagen
                img = FileIO.load(joinpath(rutaClase, archivo))
                # Redimensionar al tamaño fijo 
                img_res = imresize(img, tamanofijo)
                # Convertir a vector de Float64 y guardar
                push!(entradas, vec(Float64.(Gray.(img_res))))
                push!(etiquetas, clase)
            end
        end
    end

    # Convertimos a la matriz que esperan tus funciones 
    return stack(entradas)', etiquetas
end

# Función para cargar una imagen y prepararla
function procesarImagen(ruta::String)
    img = FileIO.load(ruta)
    # Aseguramos un tamaño fijo para que las secciones sean siempre las mismas
    img = imresize(img, (200, 200))
    return img
end

function extraerCaracteristicasSeccion(img, y_range, x_range)
    # 1. Recortamos la sección de la imagen
    seccion = img[y_range, x_range]

    # 2. Convertimos a RGB y extraemos los canales
    # channelview devuelve una matriz de (3, alto, ancho)
    canales = channelview(RGB.(seccion))

    # 3. Reorganizamos los datos a una matriz (Píxeles x 3 Canales)
    # Esto es necesario porque calculateZeroMeanNormalizationParameters espera una matriz 2D 
    pixeles = Float64.(reshape(canales, 3, :)')

    # 4. LLAMADA A TU FUNCIÓN DE solucion.jl 
    # Obtenemos las medias y desviaciones típicas de los 3 canales a la vez
    (medias, desviaciones) = calculateZeroMeanNormalizationParameters(pixeles)

    # Devolvemos un vector con los 6 valores: [R_media, G_media, B_media, R_std, G_std, B_std]
    return vcat(vec(medias), vec(desviaciones))
end

function generarDatasetCalvicieBinario(directorioBase::String, clasesInteres::Vector{String})
    entradas = []
    etiquetas = String[]
    rutas = String[] # Nuevo: para guardar la ubicación de cada foto

    for clase in clasesInteres
        rutaClase = joinpath(directorioBase, clase)
        !isdir(rutaClase) && continue

        for archivo in readdir(rutaClase)
            # Solo procesar imágenes
            if !any(ext -> endswith(lowercase(archivo), ext), [".png", ".jpg", ".jpeg"])
                continue
            end

            rutaCompleta = joinpath(rutaClase, archivo)
            img = procesarImagen(rutaCompleta)

            # Extraemos las características usando la función que definimos
            # f1 y f2 ya usan calculateZeroMeanNormalizationParameters de solucion.jl 
            f1 = extraerCaracteristicasSeccion(img, 80:120, 80:120) # Coronilla
            f2 = extraerCaracteristicasSeccion(img, 140:180, 80:120)  # Frente

            push!(entradas, vcat(f1, f2))
            push!(etiquetas, clase)
            push!(rutas, rutaCompleta) # Guardamos la ruta
        end
    end
    return Float64.(stack(entradas)'), etiquetas, rutas
end

function imprimirInformeFinal(nombreModelo::String, resultados::Tuple, clases::Vector{String})
    # Desempaquetamos las 7 métricas (cada una es una tupla media ± std) y la matriz 
    (acc, err, rec, spe, pre, npv, f1, confMat) = resultados

    println("\n" * "="^50)
    println(" RESUMEN DE EJECUCIÓN: $nombreModelo")
    println("="^50)

    println(@sprintf("Precision (Accuracy): %.4f ± %.4f", acc...))
    println(@sprintf("F1-score (Weighted):  %.4f ± %.4f", f1...))
    println(@sprintf("Sensibilidad (Recall): %.4f ± %.4f", rec...))
    println("-"^50)

    # Mejora visual de la Matriz de Confusión
    println("MATRIZ DE CONFUSIÓN ACUMULADA")
    print("\t\t")
    for c in clases
        print("$c\t")
    end
    println("\n" * "-"^50)

    for (i, clase_real) in enumerate(clases)
        print("$clase_real\t| ")
        for j in 1:length(clases)
            print("$(Int(confMat[i,j]))\t")
        end
        println()
    end
    println("="^50 * "\n")
end

# Usamos el nivel 2 porque el nivel 1 cuenta con pocas imágenes
clases_test = ["Nivel 2", "Nivel 7"]
imprimirInformeFinal("SVM Lineal (C=1.0)", resultados, clases_test)


# --- CÓDIGO FINAL DE DIAGNÓSTICO (para ver cuales son las imágenes que están fallando)
const CLASES_REALES = unique(y_raw)

let errores_totales = 0
    println("\n" * "="^40)
    println(" ANÁLISIS DE FALLOS DE LA VALIDACIÓN CRUZADA ")
    println("="^40)

    numFolds = maximum(indices_cv)

    for i in 1:numFolds
        train_idx = (indices_cv .!= i)
        test_idx = (indices_cv .== i)

        # Entrenamiento local por fold
        model_cv = SVMClassifier(kernel=LIBSVM.Kernel.Linear, cost=1.0)
        mach_cv = machine(model_cv, MLJ.table(X_norm[train_idx, :]), categorical(y_raw[train_idx]))
        MLJ.fit!(mach_cv, verbosity=0)

        # Predicción
        preds = MLJ.predict(mach_cv, MLJ.table(X_norm[test_idx, :]))
        reales_fold = y_raw[test_idx]
        rutas_fold = rutas_raw[test_idx]

        for j in 1:length(preds)
            if preds[j] != reales_fold[j]
                errores_totales += 1
                println("ERROR #$errores_totales (Fold $i)")
                println("Ruta: ", rutas_fold[j])
                println("Real: $(reales_fold[j]) | Predicho: $(preds[j])")

                # Visualización
                img_f = FileIO.load(rutas_fold[j])
                display(img_f)
                println("-"^30)
            end
        end
    end
    println("\nAnálisis finalizado. Total de fallos: $errores_totales")
end

imprimirInformeFinal("SVM Lineal (C=1.0)", resultados, CLASES_REALES)