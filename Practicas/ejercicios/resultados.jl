# resultados.jl - Experimentación con múltiples modelos para clasificación de calvicie

include("solucion.jl") # funciones de normalización, CV, métricas, etc.

using Images, FileIO, Statistics, Random, Printf, DataFrames, PrettyTables, CSV

# Fijamos la semilla para reproducibilidad
Random.seed!(1)

function procesarImagen(ruta::String)
    img = FileIO.load(ruta)
    return img
end

function extraerCaracteristicasSeccion(img, y_range, x_range)
    seccion = img[y_range, x_range]
    canales = channelview(RGB.(seccion))
    grey = channelview(Gray.(seccion))
    pixeles_rgb = Float64.(reshape(canales, 3, :)')
    pixeles_grey = Float64.(reshape(grey, 1, :)')
    medias = mean(pixeles_rgb, dims=1)
    desviaciones = std(pixeles_rgb, dims=1)
    medias2 = mean(pixeles_grey, dims=1)
    desviaciones2 = std(pixeles_grey, dims=1)
    return vcat(vec(medias), vec(desviaciones), vec(medias2), vec(desviaciones2))
end

function extraerCaracteristicasPorcentaje(img, y1, y2, x1, x2)
    alto, ancho = size(img)[1:2]
    y_range = round(Int, y1 * alto):round(Int, y2 * alto)
    x_range = round(Int, x1 * ancho):round(Int, x2 * ancho)
    return extraerCaracteristicasSeccion(img, y_range, x_range)
end

function generarDatasetCalvicieBinario(directorioBase::String, clasesInteres::Vector{String})
    entradas = []
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
            img = procesarImagen(rutaCompleta)
            zonas = vcat(
                extraerCaracteristicasPorcentaje(img, 0.675, 0.95, 0.05, 0.30),  # inferior izq
                extraerCaracteristicasPorcentaje(img, 0.675, 0.95, 0.70, 0.95),  # inferior der
                extraerCaracteristicasPorcentaje(img, 0.75, 0.95, 0.35, 0.65),   # flequillo
                extraerCaracteristicasPorcentaje(img, 0.675, 0.95, 0.30, 0.55),  # inferior centro
                extraerCaracteristicasPorcentaje(img, 0.25, 0.40, 0.005, 0.90),  # subposterior
                extraerCaracteristicasPorcentaje(img, 0.40, 0.675, 0.005, 0.90), # zona media
                extraerCaracteristicasPorcentaje(img, 0.05, 0.95, 0.05, 0.95),   # cabeza completa
                extraerCaracteristicasPorcentaje(img, 0.05, 0.25, 0.15, 0.85),   # coronilla
                extraerCaracteristicasPorcentaje(img, 0.01, 0.4, 0.05, 0.30),    # superior izq
                extraerCaracteristicasPorcentaje(img, 0.01, 0.4, 0.70, 0.95),    # superior der
                extraerCaracteristicasPorcentaje(img, 0.1, 0.9, 0.10, 0.35),     # lateral izquierdo
                extraerCaracteristicasPorcentaje(img, 0.1, 0.9, 0.65, 0.9)       # lateral derecho
            )
            push!(entradas, zonas)
            push!(etiquetas, clase)
            push!(rutas, rutaCompleta)
        end
    end
    return Float64.(stack(entradas)'), etiquetas, rutas
end
# ------------------------------------------------------------

# ------------------------------------------------------------
# CONFIGURACIÓN DEL EXPERIMENTO
# ------------------------------------------------------------
# Ruta al dataset
ruta_dataset = joinpath(@__DIR__, "../../Dataset")

# Seleccionamos las clases
#clases_todas = ["nivel_$i" for i in 1:7]
clases_todas = ["nivel_1","nivel_3","nivel_5","nivel_7"]

println("Cargando dataset desde: $ruta_dataset")
X_raw, y_raw, _ = generarDatasetCalvicieBinario(ruta_dataset, clases_todas)
println("Instancias cargadas: $(size(X_raw,1)), características: $(size(X_raw,2))")

# Normalización MinMax (usando función de solucion.jl)
X_norm = normalizeMinMax(X_raw)

# Validación cruzada estratificada (10 folds) usando función de solucion.jl
indices_cv = crossvalidation(y_raw, 10)
println("Validación cruzada con $(maximum(indices_cv)) folds")

# ------------------------------------------------------------
# DEFINICIÓN DE MODELOS A PROBAR
# ------------------------------------------------------------

# 1. SVM: 8 configuraciones (lineal C=0.1,1,10; RBF C=1,10; polinomial grado2 y 3; sigmoide)
modelos_svm = [
    ("SVM Lineal C=0.1", :SVC, Dict("kernel"=>"linear", "C"=>0.1)),
    ("SVM Lineal C=1.0", :SVC, Dict("kernel"=>"linear", "C"=>1.0)),
    ("SVM Lineal C=10",  :SVC, Dict("kernel"=>"linear", "C"=>10)),
    ("SVM RBF C=1.0",    :SVC, Dict("kernel"=>"rbf",    "C"=>1.0)),
    ("SVM RBF C=10",     :SVC, Dict("kernel"=>"rbf",    "C"=>10)),
    ("SVM Polinomial g2 C=1", :SVC, Dict("kernel"=>"poly", "C"=>1.0, "degree"=>2)),
    ("SVM Polinomial g3 C=1", :SVC, Dict("kernel"=>"poly", "C"=>1.0, "degree"=>3)),
    ("SVM Sigmoid C=1",       :SVC, Dict("kernel"=>"sigmoid", "C"=>1.0))
]

# 2. kNN: 6 valores de k (1,3,5,7,9,11)
modelos_knn = [
    ("k-NN k=1",  :KNeighborsClassifier, Dict("n_neighbors"=>1)),
    ("k-NN k=3",  :KNeighborsClassifier, Dict("n_neighbors"=>3)),
    ("k-NN k=5",  :KNeighborsClassifier, Dict("n_neighbors"=>5)),
    ("k-NN k=7",  :KNeighborsClassifier, Dict("n_neighbors"=>7)),
    ("k-NN k=9",  :KNeighborsClassifier, Dict("n_neighbors"=>9)),
    ("k-NN k=11", :KNeighborsClassifier, Dict("n_neighbors"=>11))
]

# 3. Árboles de decisión: 6 profundidades (2,4,6,8,10,12)
modelos_dt = [
    ("Árbol prof. 2",  :DecisionTreeClassifier, Dict("max_depth"=>2)),
    ("Árbol prof. 4",  :DecisionTreeClassifier, Dict("max_depth"=>4)),
    ("Árbol prof. 6",  :DecisionTreeClassifier, Dict("max_depth"=>6)),
    ("Árbol prof. 8",  :DecisionTreeClassifier, Dict("max_depth"=>8)),
    ("Árbol prof. 10", :DecisionTreeClassifier, Dict("max_depth"=>10)),
    ("Árbol prof. 12", :DecisionTreeClassifier, Dict("max_depth"=>12))
]

# 4. RNA: 8 arquitecturas (1 o 2 capas ocultas)
#    Se usa :ANN, y los hiperparámetros van en un Dict con "topology", "maxEpochs", etc.
topologias = [
    [4], [8], [16],          # 1 capa oculta
    [4,4], [8,4], [16,8],    # 2 capas ocultas
    [32], [32,16]            # más neuronas
]
modelos_ann = []
for topo in topologias
    nombre = "RNA " * join(topo, "-")
    # Parámetros: topology, maxEpochs, learningRate, numExecutions (repeticiones por fold)
    hiper = Dict("topology"=>topo, "maxEpochs"=>200, "learningRate"=>0.01, "numExecutions"=>3)
    push!(modelos_ann, (nombre, :ANN, hiper))
end

# 5. DoME: 8 valores de maximumNodes (5,10,15,20,25,30,35,40)
modelos_dome = []
for nodes in [5,10,15,20,25,30,35,40]
    nombre = "DoME maxNodes=$nodes"
    push!(modelos_dome, (nombre, :DoME, Dict("maximumNodes"=>nodes)))
end

# Unimos todos en una sola lista
modelos = vcat(modelos_svm, modelos_knn, modelos_dt, modelos_ann, modelos_dome)

println("Total de configuraciones a probar: $(length(modelos))")

# Almacenamos resultados
resultados = []

println("\n" * "="^70)
println("INICIANDO EXPERIMENTOS (esto puede tomar unos minutos)...")
println("="^70)

for (nombre, tipo, hiper) in modelos
    println("Probando: $nombre ...")
    # Llamamos a modelCrossValidation (definida en solucion.jl)
    # Devuelve 7 tuplas y la matriz de confusión: (acc, err, rec, spe, pre, npv, f1, confMat)
    (acc, err, rec, spe, pre, npv, f1, confMat) = modelCrossValidation(tipo, hiper, (X_norm, y_raw), indices_cv)
    # Guardamos nombre, accuracy (media, std), f1, recall y matriz
    push!(resultados, (nombre, acc[1], acc[2], f1[1], f1[2], rec[1], rec[2], confMat))
end

# ------------------------------------------------------------
# TABLA DE RESULTADOS
# ------------------------------------------------------------
using DataFrames, PrettyTables

df = DataFrame(
    Modelo = [r[1] for r in resultados],
    Accuracy = [@sprintf("%.4f ± %.4f", r[2], r[3]) for r in resultados],
    F1_score = [@sprintf("%.4f ± %.4f", r[4], r[5]) for r in resultados],
    Recall   = [@sprintf("%.4f ± %.4f", r[6], r[7]) for r in resultados]
)

println("\n TABLA COMPARATIVA DE MODELOS")
PrettyTables.pretty_table(df, alignment = :l, border_crayon = crayon"yellow")

# Guardar a CSV
CSV.write("resultados_experimentos.csv", df)
println("\n Tabla guardada en 'resultados_experimentos.csv'")

# ------------------------------------------------------------
# MEJOR MODELO (mayor accuracy media)
# ------------------------------------------------------------
best_idx = argmax([r[2] for r in resultados])  # r[2] es mean accuracy
best_name = resultados[best_idx][1]
best_acc_mean = resultados[best_idx][2]
best_acc_std = resultados[best_idx][3]
best_f1_mean = resultados[best_idx][4]
best_f1_std = resultados[best_idx][5]
best_rec_mean = resultados[best_idx][6]
best_rec_std = resultados[best_idx][7]
best_confMat = resultados[best_idx][8]

println("\n" * "="^70)
println("MEJOR MODELO: $best_name")
println("   Accuracy: $(@sprintf("%.4f ± %.4f", best_acc_mean, best_acc_std))")
println("   F1-score: $(@sprintf("%.4f ± %.4f", best_f1_mean, best_f1_std))")
println("   Recall:   $(@sprintf("%.4f ± %.4f", best_rec_mean, best_rec_std))")
println("="^70)

# Mostrar matriz de confusión del mejor modelo (acumulada en validación cruzada)
clases_ordenadas = unique(y_raw)
println("\nMATRIZ DE CONFUSIÓN (acumulada en 10 folds) - $best_name")
println("Filas = clase real, Columnas = clase predicha\n")

# Cabecera
print(" " ^ 12)
for c in clases_ordenadas
    print(rpad(c, 12))
end
println()
println("-" ^ (12 + 12 * length(clases_ordenadas)))

for (i, real) in enumerate(clases_ordenadas)
    print(rpad(real, 12))
    for j in 1:length(clases_ordenadas)
        print(rpad(round(Int, best_confMat[i,j]), 12))
    end
    println()
end

println("\n(Nota: la matriz está en formato \"filas reales, columnas predichas\")")
