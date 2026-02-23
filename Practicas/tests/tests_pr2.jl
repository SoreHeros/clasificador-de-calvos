using Statistics
using Flux
using Test
using DelimitedFiles
using Random: seed!

include("../ejercicios/pr2.jl")

println("="^70)
println("EJECUTANDO TESTS PROPIOS")
println("="^70)

@testset "Tests Propios" begin

    @testset "oneHotEncoding" begin
        r = oneHotEncoding(["a", "b", "c", "a"], ["a", "b", "c"])
        @test size(r) == (4, 3)
        @test sum(r, dims=2) == ones(4, 1)

        r2 = oneHotEncoding(["si", "no", "si"])
        @test size(r2, 2) == 1

        r3 = oneHotEncoding([true, false, true])
        @test r3 == reshape([true, false, true], :, 1)

        r4 = oneHotEncoding(String[], String[])
        @test size(r4) == (0, 0)

        r5 = oneHotEncoding([1, 2, 1])
        @test size(r5) == (3, 1)

        r6 = oneHotEncoding([1, 2, 3])
        @test size(r6) == (3, 3)
    end

    @testset "normalizacion" begin
        data = Float64[0 0; 5 10; 10 20]

        norm_data = normalizeMinMax(data)
        @test minimum(norm_data) >= 0
        @test maximum(norm_data) <= 1

        norm_data2 = normalizeZeroMean(data)
        @test isapprox(mean(norm_data2, dims=1), [0 0], atol=1e-10)

        data2 = ones(2, 2)
        norm_mm = normalizeMinMax(data2)
        @test all(norm_mm .== 0)

        norm_zm = normalizeZeroMean(data2)
        @test all(norm_zm .== 0)

        data3 = Float64[-1; 1]
        norm_mm3 = normalizeMinMax(reshape(data3, :, 1))
        @test norm_mm3 ≈ reshape([0; 1], :, 1)

        norm_zm3 = normalizeZeroMean(reshape(data3, :, 1))
        @test mean(norm_zm3) ≈ 0
        @test std(norm_zm3) ≈ 1
    end

    @testset "Clasificacion y Accuracy" begin
        out = [0.2, 0.6, 0.8]
        @test classifyOutputs(out) == [false, true, true]

        targets = [false, true, false]
        @test accuracy(classifyOutputs(out), targets) ≈ 2 / 3

        out_multi = [0.1 0.9; 0.8 0.2]
        class_multi = classifyOutputs(out_multi)
        @test class_multi == [false true; true false]

        acc_real = accuracy([0.4, 0.6], [false, true])
        @test acc_real == 1.0

        acc_wrong = accuracy([true, false], [false, true])
        @test acc_wrong == 0.0

        out_tie = [0.5 0.5]
        @test classifyOutputs(out_tie)[1, :] == [true, false]
    end

    @testset "buildClassANN" begin
        ann = buildClassANN(2, [4], 1)
        x = rand(Float32, 2, 5)
        y = ann(x)
        @test size(y) == (1, 5)

        ann_empty = buildClassANN(3, Int[], 2)
        y_empty = ann_empty(rand(Float32, 3, 1))
        @test size(y_empty) == (2, 1)
        @test sum(y_empty) ≈ 1

        ann_relu = buildClassANN(2, [3], 1; transferFunctions=[relu])
        @test true

        @test_throws AssertionError buildClassANN(2, [3], 1; transferFunctions=Function[])
    end


    @testset "trainClassANN" begin
        inputs = rand(Float32, 10, 2)
        targets = rand(Bool, 10, 1)
        dataset = (inputs, targets)
        topology = [4]

        # --- Prueba de Ciclo 0 y Longitud ---
        ann, losses = trainClassANN(topology, dataset; maxEpochs=10, minLoss=-1.0)

        # El vector de pérdidas debe tener n+1 elementos (0 a n)
        @test length(losses) == 11
        # El primer valor debe ser mayor que 0 (error inicial)
        @test losses[1] > 0

        # --- Prueba de Tipado ---
        # Comprobamos que los datos esten en Float32 como pide el enunciado
        @test eltype(losses) == Float32

        # --- Prueba de Reproducibilidad ---
        # Random.seed!(1) asegurar que dos llamadas produzcan lo mismo
        ann1, losses1 = trainClassANN(topology, dataset; maxEpochs=5, minLoss=-1.0)
        ann2, losses2 = trainClassANN(topology, dataset; maxEpochs=5, minLoss=-1.0)

        @test losses1 == losses2
        # Verificar pesos de la primera capa para asegurar que la red es idéntica
        @test ann1[1].weight == ann2[1].weight

        # --- Prueba de Transposición y Entrenamiento ---
        # Si funciona sin error de dimensión, la transposición es coherente.
        ann_train, losses_train = trainClassANN(topology, dataset; maxEpochs=100, learningRate=0.05, minLoss=-1.0)
        # El error final debería ser menor o igual al inicial (generalmente menor)
        @test losses_train[end] <= losses_train[1]

        # --- Prueba de Parada Temprana (minLoss) ---
        # Entrenamos con un minLoss alto que sea fácil de alcanzar
        # Como los inputs son aleatorios, el error no bajará a 0, pero si ponemos un minLoss
        # mayor que el error inicial, debería parar en el ciclo 0 o 1.
        # Para asegurar, usamos un threshold alto.
        initial_loss = losses[1]
        target_min_loss = initial_loss * 1.5 # Imposible de bajar si es Loss, pero si es mayor... 
        # Espera, minLoss es una cota INFERIOR. Si currentLoss < minLoss para.
        # Para probar esto, ne  cesitamos que el error baje lo suficiente.
        # O podemos poner un minLoss que sepamos que se alcanzará.
        # Mejor estrategia: Entrenar X épocas, ver el loss final. Entrenar de nuevo con minLoss = loss final + epsilon.
        # Debería parar antes.

        ann_early, losses_early = trainClassANN(topology, dataset; maxEpochs=1000, minLoss=losses_train[end] + 0.05)
        # Debería haber parado antes de las 1000 épocas si alcanzó el minLoss
        @test length(losses_early) < 1001

        # --- Prueba de Vectores ---
        targets_vec = vec(targets)
        dataset_vec = (inputs, targets_vec)

        ann_vec, losses_vec = trainClassANN(topology, dataset_vec; maxEpochs=5, minLoss=-1.0)

        @test length(losses_vec) == 6
        @test losses_vec[1] > 0
        @test eltype(losses_vec) == Float32
    end

    @testset "Estructura y Tipado de trainClassANN" begin
        # Datos de prueba mínimos
        X = rand(10, 2)
        Y = rand(Bool, 10, 2)
        topology = [3]

        # Caso 1: Solo entrenamiento (Val y Test vacíos)
        (ann, tr_L, val_L, te_L) = trainClassANN(topology, (X, Y); maxEpochs=5)

        @test length(tr_L) == 6 # n+1 elementos (Ciclo 0 + 5 épocas) 
        @test isempty(val_L) && isempty(te_L)
        @test eltype(tr_L) == Float32

        # Caso 2: Con todos los conjuntos
        Xv = rand(5, 2)
        Yv = rand(Bool, 5, 2)
        Xt = rand(5, 2)
        Yt = rand(Bool, 5, 2)
        (ann, tr_L, val_L, te_L) = trainClassANN(topology, (X, Y);
            validationDataset=(Xv, Yv), testDataset=(Xt, Yt), maxEpochs=2)

        @test length(tr_L) == 3
        @test length(val_L) == 3
        @test length(te_L) == 3
    end

    @testset "Lógica de Parada Temprana y Mejor RNA" begin
        # Usamos datos de Iris para una prueba real
        dataset_iris = readdlm(joinpath(@__DIR__, "iris.data"), ',')
        inputs_iris = convert(Array{Float32,2}, dataset_iris[:, 1:4])
        targets_iris = oneHotEncoding(dataset_iris[:, 5])
        normalizeMinMax!(inputs_iris)

        # Forzamos un caso donde la red NO mejora (Validación con mucho ruido)
        # El entrenamiento debería parar en el ciclo 0 + maxEpochsVal
        max_val = 5
        seed!(1)
        (ann, tr_L, val_L, _) = trainClassANN([2], (inputs_iris, targets_iris);
            validationDataset=(rand(Float32, 50, 4), rand(Bool, 50, 3)),
            maxEpochs=100, maxEpochsVal=max_val)

        @test length(val_L) <= (max_val + 2) # Parada temprana detectada (allow 1 spurious improvement)
    end

    @testset "Versión para Vectores (Binary Classification)" begin
        X = rand(Float32, 10, 2)
        Y_vec = rand(Bool, 10) # Vector de 1D

        # Debe funcionar sin errores de dimensión gracias al reshape interno
        results = trainClassANN([2], (X, Y_vec); maxEpochs=2)
        @test length(results) == 4
        @test size(results[1](X'[:, 1])) == (1,) # Salida de 1 neurona
    end

    # ==================== NUEVOS TESTS PARA EJERCICIO 4 ====================
    @testset "confusionMatrix - Binario 1D (Boolean)" begin
        # Test 1: Vector booleano
        outputs = Bool[1, 1, 0, 0, 1, 0]
        targets = Bool[1, 0, 1, 0, 1, 1]

        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
        
        # Verificar métricas
        @test isapprox(acc, 0.5)
        @test isapprox(errorRate, 0.5)
        @test isapprox(recall, 0.5)
        @test isapprox(specificity, 0.5)
        @test isapprox(precision, 2/3)
        @test isapprox(NPV, 1/3)
        @test isapprox(F1, 2*(2/3*0.5)/(2/3+0.5))
        
        # Matriz de confusión según PDF: [TN FP; FN TP]
        # TN = 1, FP = 1, FN = 2, TP = 2
        expected_matrix = [1 1; 2 2]  # [TN FP; FN TP]
        @test confMatrix == expected_matrix
        
        # Test 2: Todos correctos
        outputs = Bool[1, 1, 0, 0]
        targets = Bool[1, 1, 0, 0]
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
        
        @test isapprox(acc, 1.0)
        @test isapprox(errorRate, 0.0)
        @test confMatrix == [2 0; 0 2]  # [TN=2, FP=0; FN=0, TP=2]
        
        # Test 3: Todos incorrectos
        outputs = Bool[1, 1, 0, 0]
        targets = Bool[0, 0, 1, 1]
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
        
        @test isapprox(acc, 0.0)
        @test isapprox(errorRate, 1.0)
        @test confMatrix == [0 2; 2 0]  # [TN=0, FP=2; FN=2, TP=0]
        
        println(" ✓ confusionMatrix binario 1D (Boolean) correcto")
    end

    @testset "confusionMatrix - Binario 1D (Real con threshold)" begin
        # Test con threshold por defecto (0.5)
        outputs = [0.2, 0.7, 0.3, 0.9, 0.4]
        targets = Bool[1, 1, 0, 1, 0]
        
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
        
        @test isapprox(acc, 0.8)
        @test isapprox(recall, 2/3)
        @test isapprox(specificity, 1.0)
        @test confMatrix == [2 0; 1 2]  # [TN=2, FP=0; FN=1, TP=2]
        
        # Test con threshold que cambia clasificación (0.8)
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; threshold=0.8)
        
        # Clasificados: [0,0,0,1,0]
        # TP: 1, FN: 2, TN: 2, FP: 0
        expected2 = [2 0; 2 1]  # [TN=2, FP=0; FN=2, TP=1]
        
        @test isapprox(acc, 0.6)
        @test confMatrix == expected2
        
        println(" ✓ confusionMatrix binario 1D (Real con threshold) correcto")
    end

    @testset "confusionMatrix - Binario 2D" begin
        # Caso binario (1 columna)
        outputs = Bool[1; 1; 0; 0; 1; 0]  # Matriz 6x1
        targets = Bool[1; 0; 1; 0; 1; 1]  # Matriz 6x1
        
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets)
        
        @test isapprox(acc, 0.5)
        @test confMatrix == [1 1; 2 2]  # [TN=1, FP=1; FN=2, TP=2]
        
        println(" ✓ confusionMatrix binario 2D correcto")
    end
    
    @testset "confusionMatrix - Multiclase 2D (Boolean)" begin
        # 9 patrones, 3 clases
        outputs = Bool[
            1 0 0;  # patrón 1: clase 1
            1 0 0;  # patrón 2: clase 1
            1 0 0;  # patrón 3: clase 1
            0 1 0;  # patrón 4: clase 2
            0 1 0;  # patrón 5: clase 2
            0 1 0;  # patrón 6: clase 2
            0 0 1;  # patrón 7: clase 3
            0 0 1;  # patrón 8: clase 3
            0 0 1   # patrón 9: clase 3
        ]
        
        targets = Bool[
            1 0 0;  # patrón 1: clase 1
            0 1 0;  # patrón 2: clase 2
            0 0 1;  # patrón 3: clase 3
            1 0 0;  # patrón 4: clase 1
            0 1 0;  # patrón 5: clase 2
            0 0 1;  # patrón 6: clase 3
            1 0 0;  # patrón 7: clase 1
            0 1 0;  # patrón 8: clase 2
            0 0 1   # patrón 9: clase 3
        ]
        
        # Test sin normalizar (weighted=true - por defecto)
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=true)
        
        expected_matrix = [
            1 1 1;
            1 1 1;
            1 1 1
        ]
        
        @test isapprox(acc, 3/9)  # 3 correctos / 9 total = 1/3
        @test isapprox(errorRate, 6/9)  # 6 incorrectos / 9 total = 2/3
        @test isapprox(recall, 1/3)  # Media ponderada de recalls por clase
        @test isapprox(specificity, 2/3)
        @test isapprox(precision, 1/3)
        @test isapprox(NPV, 2/3)
        @test isapprox(F1, 1/3)
        @test confMatrix == expected_matrix
        
        # Test con normalización por filas (weighted=false)
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix_norm) = confusionMatrix(outputs, targets; weighted=false)
        
        # La matriz normalizada debe tener 1/3 en cada posición
        expected_norm = [
            1/3 1/3 1/3;
            1/3 1/3 1/3;
            1/3 1/3 1/3
        ]
        
        @test all(isapprox.(confMatrix_norm, expected_norm, atol=1e-10))
        
        println(" ✓ confusionMatrix multiclase 2D (Boolean) correcto")
    end

    @testset "confusionMatrix - Multiclase 2D (Real con threshold)" begin
        # 9 patrones, 3 clases con valores reales (probabilidades)
        outputs = Float64[
            1.2 0.1 0.1;  # patrón 1: clase 1 (más alta)
            1.1 0.2 0.3;  # patrón 2: clase 1
            1.0 0.5 0.5;  # patrón 3: clase 1
            0.2 1.1 0.3;  # patrón 4: clase 2
            0.3 1.2 0.1;  # patrón 5: clase 2
            0.4 1.0 0.6;  # patrón 6: clase 2
            0.1 0.2 1.3;  # patrón 7: clase 3
            0.2 0.3 1.1;  # patrón 8: clase 3
            0.3 0.4 1.0   # patrón 9: clase 3
        ]
        
        targets = Bool[
            1 0 0;  # patrón 1: clase 1
            0 1 0;  # patrón 2: clase 2
            0 0 1;  # patrón 3: clase 3
            1 0 0;  # patrón 4: clase 1
            0 1 0;  # patrón 5: clase 2
            0 0 1;  # patrón 6: clase 3
            1 0 0;  # patrón 7: clase 1
            0 1 0;  # patrón 8: clase 2
            0 0 1   # patrón 9: clase 3
        ]
        
        # Test con threshold por defecto (usando argmax)
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=true)
        
        expected_matrix = [
            1 1 1;
            1 1 1;
            1 1 1
        ]
        
        @test isapprox(acc, 3/9)
        @test isapprox(errorRate, 6/9)
        @test confMatrix == expected_matrix
        
        println(" ✓ confusionMatrix multiclase 2D (Real con threshold) correcto")
    end

    @testset "confusionMatrix - Vectores de Any (con clases explícitas)" begin
        # Datos de ejemplo con strings
        outputs = ["gato", "perro", "pájaro", "gato", "perro", "pájaro", "gato", "perro", "pájaro"]
        targets = ["gato", "perro", "pájaro", "perro", "pájaro", "gato", "pájaro", "gato", "perro"]
        classes = ["gato", "perro", "pájaro"]
        
        # Test sin normalizar
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets, classes; weighted=true)
        
        expected_matrix = [
            1 1 1;
            1 1 1;
            1 1 1
        ]
        
        @test isapprox(acc, 3/9)
        @test isapprox(errorRate, 6/9)
        @test confMatrix == expected_matrix
        
        # Test con normalización
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix_norm) = confusionMatrix(outputs, targets, classes; weighted=false)
        
        expected_norm = [
            1/3 1/3 1/3;
            1/3 1/3 1/3;
            1/3 1/3 1/3
        ]
        
        @test all(isapprox.(confMatrix_norm, expected_norm, atol=1e-10))
        
        println(" ✓ confusionMatrix vectores Any (con clases explícitas) correcto")
    end

    @testset "confusionMatrix - Vectores de Any (clases automáticas)" begin
        # Datos de ejemplo con strings (clases se extraen automáticamente)
        outputs = ["gato", "perro", "pájaro", "gato", "perro", "pájaro", "gato", "perro", "pájaro"]
        targets = ["gato", "perro", "pájaro", "perro", "pájaro", "gato", "pájaro", "gato", "perro"]
        
        # Test sin normalizar
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=true)
        
        expected_matrix = [
            1 1 1;
            1 1 1;
            1 1 1
        ]
        
        @test isapprox(acc, 3/9)
        @test isapprox(errorRate, 6/9)
        @test confMatrix == expected_matrix
        @test size(confMatrix) == (3, 3)
        
        println(" ✓ confusionMatrix vectores Any (clases automáticas) correcto")
    end
end

println("\n✓ Todos los tests propios completados correctamente")


###### RUBRICA OFICIAL ####

println("\n" * "="^70)
println("EJECUTANDO TESTS DE LA RUBRICA OFICIAL")
println("="^70)

# Cargamos el dataset Iris
dataset = readdlm(joinpath(@__DIR__, "iris.data"), ',')
inputs = convert(Array{Float32,2}, dataset[:, 1:4])

@testset "Rubrica Oficial" begin

    @testset "oneHotEncoding con Iris" begin
        targets = oneHotEncoding(dataset[:, 5])

        @test size(targets) == (150, 3)
        @test all(targets[1:50, 1]) && !any(targets[1:50, 2:3])  # Primera clase
        @test all(targets[51:100, 2]) && !any(targets[51:100, [1, 3]])  # Segunda clase
        @test all(targets[101:150, 3]) && !any(targets[101:150, 1:2])  # Tercera clase

        println("   ✓ oneHotEncoding con dataset Iris correcto")
    end

    @testset "Normalizacion con Iris" begin
        newInputs = normalizeMinMax(inputs)
        @test !isnothing(newInputs)
        @test all(minimum(newInputs, dims=1) .== 0)
        @test all(maximum(newInputs, dims=1) .== 1)
        println("   ✓ normalizeMinMax correcto")

        # Normalizacion entre maximo y minimo
        newInputs = normalizeZeroMean(inputs)
        @test !isnothing(newInputs)
        @test all(abs.(mean(newInputs, dims=1)) .<= 1e-4)
        @test all(isapprox.(std(newInputs, dims=1), 1))
        println("   ✓ normalizeZeroMean correcto")

        # Normalizacion de media 0
        normalizeMinMax!(inputs)
        @test all(minimum(inputs, dims=1) .== 0)
        @test all(maximum(inputs, dims=1) .== 1)
        println("   ✓ normalizeMinMax! correcto")
    end

    @testset "classifyOutputs" begin
        @test classifyOutputs(0.1:0.1:1; threshold=0.65) == [falses(6); trues(4)]
        @test classifyOutputs([1 2 3; 3 2 1; 2 3 1; 2 1 3]) == Bool[0 0 1; 1 0 0; 0 1 0; 0 0 1]
        println("   ✓ classifyOutputs correcto")
    end

    # Creacion de RNA correcta
    @testset "buildClassANN con Iris" begin
        ann = buildClassANN(4, [5], 3)

        @test length(ann) == 3
        @test ann[3] == softmax
        @test size(ann[1].weight) == (5, 4)
        @test size(ann[2].weight) == (3, 5)
        @test size(ann(inputs')) == (3, 150)
        println("   ✓ buildClassANN correcto")
    end

    # Comprobar accuracy 
    @testset "accuracy con Iris" begin
        targets = oneHotEncoding(dataset[:, 5])

        @test isapprox(accuracy([sin.(1:150) cos.(1:150) tan.(1:150)], targets), 0.34)
        @test isapprox(accuracy(1:(-1/150):(1/150), targets[:, 1]; threshold=0.75), 0.92)
        println("   ✓ accuracy correcto")
    end

    # Comprobar la funcion hold out  
    @testset "hold out" begin
        # Establecemos la semilla para que los resultados sean siempre los mismos
        # Comprobamos que la generación de números aleatorios es la esperada:
        seed!(1)
        @assert(isapprox(rand(), 0.07336635446929285))
        #  Si fallase aquí, seguramente dara error al comprobar los resultados de la ejecución de la siguiente función porque depende de la generación de números aleatorios


        # Unas comprobaciones sencillas de la función holdOut:
        @assert(all(length.(holdOut(10, 0.3)) .== [7, 3]))
        @assert(all(length.(holdOut(10, 0.3, 0.2)) .== [5, 3, 2]))
        println("   ✓ hold out correcto")

    end

    targets = oneHotEncoding(dataset[:, 5])

    @testset "trainClassANN" begin
        # CASO 1: Comprobación del Ciclo 0 (maxEpochs=0)
        # Se verifica que los vectores de loss se inicializan correctamente antes de entrenar
        seed!(1)
        (ann, trainingLosses, validationLosses, testLosses) = trainClassANN([4, 3], (inputs, targets);
            validationDataset=(inputs[101:150, :], targets[101:150, :]),
            testDataset=(inputs[51:100, :], targets[51:100, :]),
            maxEpochs=0, maxEpochsVal=5)

        @assert(all(isapprox.(trainingLosses, Float32[1.2139437]; rtol=1e-5)))
        @assert(all(isapprox.(validationLosses, Float32[1.1530712]; rtol=1e-5)))
        @assert(all(isapprox.(testLosses, Float32[1.8724904]; rtol=1e-5)))


        # CASO 2: Parada Temprana por Validación (Early Stopping)
        # Se fuerza el solapamiento para que el error de validación suba y el entrenamiento pare
        seed!(1)
        (ann, trainingLosses, validationLosses, testLosses) = trainClassANN([4, 3], (inputs, targets);
            validationDataset=(inputs[101:150, :], targets[101:150, :]),
            testDataset=(inputs[51:100, :], targets[51:100, :]),
            maxEpochs=100, maxEpochsVal=5)

        result_trainingLosses = Float32[1.2139437, 1.2000579, 1.1873707, 1.1757828, 1.165133, 1.155307, 1.1462542, 1.1379561, 1.1304016, 1.1235778, 1.117466, 1.1120408]
        result_validationLosses = Float32[1.1530712, 1.1259663, 1.1016681, 1.0822797, 1.0691929, 1.0617257, 1.0585984, 1.0587119, 1.061241, 1.065562, 1.0711765, 1.0776592]
        result_testLosses = Float32[1.8724904, 1.8293855, 1.787262, 1.7450062, 1.7017598, 1.6577507, 1.6135957, 1.569862, 1.5269873, 1.4853015, 1.4450579, 1.4064597]

        @assert(all(isapprox.(trainingLosses, result_trainingLosses; rtol=1e-5)))
        @assert(all(isapprox.(validationLosses, result_validationLosses; rtol=1e-5)))
        @assert(all(isapprox.(testLosses, result_testLosses; rtol=1e-5)))


        # CASO 3: Sin mejora (la RNA inicial es la mejor)
        # Se repiten patrones para forzar que el error de validación nunca mejore desde el ciclo 0
        seed!(1)
        (ann, trainingLosses, validationLosses, testLosses) = trainClassANN([4, 3], (inputs, targets);
            validationDataset=(repeat(inputs[[1], :], 100, 1), repeat(targets[[1], :], 100, 1)),
            testDataset=(repeat(inputs[[101], :], 100, 1), repeat(targets[[101], :], 100, 1)),
            maxEpochs=100, maxEpochsVal=5)

        result_trainingLosses = Float32[1.2139437, 1.2000579, 1.1873707, 1.1757828, 1.165133, 1.155307]
        result_validationLosses = Float32[0.6153051, 0.64382416, 0.6721526, 0.6990008, 0.72334856, 0.74530566]
        result_testLosses = Float32[1.1561297, 1.1288316, 1.1043872, 1.084908, 1.0717835, 1.0643175]

        @assert(all(isapprox.(trainingLosses, result_trainingLosses; rtol=1e-5)))
        @assert(all(isapprox.(validationLosses, result_validationLosses; rtol=1e-5)))
        @assert(all(isapprox.(testLosses, result_testLosses; rtol=1e-5)))


        # CASO 4: Salidas deseadas como vectores (2 clases / reshape interno)
        seed!(1)
        (ann, trainingLosses, validationLosses, testLosses) = trainClassANN([4, 3], (inputs, targets[:, 1]);
            validationDataset=(inputs[51:100, :], targets[51:100, 2]),
            testDataset=(inputs, targets[:, 3]),
            maxEpochs=100, maxEpochsVal=5)

        result_trainingLosses = Float32[0.7061655, 0.6974345, 0.68925875, 0.6816472, 0.67460465, 0.6681309]
        result_validationLosses = Float32[0.6585732, 0.6816271, 0.7051577, 0.7291063, 0.75340605, 0.7779797]
        result_testLosses = Float32[0.7051467, 0.69689614, 0.6892454, 0.6822056, 0.67578304, 0.66998]

        @assert(all(isapprox.(trainingLosses, result_trainingLosses; rtol=1e-5)))
        @assert(all(isapprox.(validationLosses, result_validationLosses; rtol=1e-5)))
        @assert(all(isapprox.(testLosses, result_testLosses; rtol=1e-5)))
        println("   ✓ trainClassANN correcto")
    end

    # ==================== TESTS DEL PROFESOR PARA EJERCICIO 4 ====================
    @testset "confusionMatrix - Tests del profesor" begin
        # Test 1: Binario con vector booleano
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
            confusionMatrix(sin.(1:8).>=0, [falses(4); trues(4)])
        
        @test isapprox(acc, 0.375)
        @test isapprox(errorRate, 1-0.375)
        @test isapprox(recall, 0.5)
        @test isapprox(specificity, 0.25)
        @test isapprox(precision, 0.4)
        @test isapprox(NPV, 1/3.)
        @test isapprox(F1, 4/9.)
        @test confMatrix == [1 3; 2 2]
        
        # Test 2: Binario con vector real y threshold
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
            confusionMatrix(sin.(1:8), [falses(4); trues(4)]; threshold=0.9)
        
        @test isapprox(acc, 0.5)
        @test isapprox(errorRate, 0.5)
        @test isapprox(recall, 0.25)
        @test isapprox(specificity, 0.75)
        @test isapprox(precision, 0.5)
        @test isapprox(NPV, 0.5)
        @test isapprox(F1, 1/3.)
        @test confMatrix == [3 1; 3 1]
        
        # Test 3: Multiclase con matrices booleanas (9 patrones, 3 clases)
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
            confusionMatrix(
                Bool[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1],
                Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1];
                weighted=true)
        
        @test isapprox(acc, 1/3.)
        @test isapprox(errorRate, 2/3.)
        @test isapprox(recall, 1/3.)
        @test isapprox(specificity, 2/3.)
        @test isapprox(precision, 1/3.)
        @test isapprox(NPV, 2/3.)
        @test isapprox(F1, 1/3.)
        @test confMatrix == [1 1 1; 1 1 1; 1 1 1]
        
        # Test 4: Multiclase con matrices reales
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
            confusionMatrix(
                Float64[1 0 0; 1 0 0; 1 0 0; 0 1 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1; 0 0 1] .+ 1,
                Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0; 0 0 1];
                weighted=true)
        
        @test isapprox(acc, 1/3.)
        @test isapprox(errorRate, 2/3.)
        @test isapprox(recall, 1/3.)
        @test isapprox(specificity, 2/3.)
        @test isapprox(precision, 1/3.)
        @test isapprox(NPV, 2/3.)
        @test isapprox(F1, 1/3.)
        @test confMatrix == [1 1 1; 1 1 1; 1 1 1]
        
        # Test 5: Con dataset Iris (vectores de strings)
        dataset = readdlm(joinpath(@__DIR__, "iris.data"), ',')
        targets = dataset[:, 5]
        (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = 
            confusionMatrix(repeat(unique(targets), 50), targets)
        
        @test isapprox(acc, 1/3.)
        @test isapprox(errorRate, 2/3.)
        @test isapprox(recall, 1/3.)
        @test isapprox(specificity, 2/3.)
        @test isapprox(precision, 1/3.)
        @test isapprox(NPV, 2/3.)
        @test isapprox(F1, 1/3.)
        @test confMatrix == [17 17 16; 17 16 17; 16 17 17]
        
        println(" ✓ confusionMatrix - Todos los tests del profesor correctos")
    end
end

println("\n" * "="^70)
println("✓✓✓ TODOS LOS TESTS COMPLETADOS CORRECTAMENTE ✓✓✓")
println("="^70)
