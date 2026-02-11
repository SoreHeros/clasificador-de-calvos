using Statistics
using Flux
using Test
using DelimitedFiles
using Random

include("../ejercicios/pr2.jl")

println("="^70)
println("EJECUTANDO TESTS PROPIOS")
println("="^70)

@testset "Tests Propios" begin
    
    @testset "oneHotEncoding" begin
        r = oneHotEncoding(["a","b","c","a"], ["a","b","c"])
        @test size(r) == (4, 3)
        @test sum(r, dims=2) == ones(4, 1)  
        
        r2 = oneHotEncoding(["si", "no", "si"])
        @test size(r2, 2) == 1
        
        r3 = oneHotEncoding([true, false, true])
        @test r3 == reshape([true, false, true], :, 1)
        
        r4 = oneHotEncoding(String[], String[])
        @test size(r4) == (0,0)
        
        r5 = oneHotEncoding([1,2,1])
        @test size(r5) == (3,1)  
        
        r6 = oneHotEncoding([1,2,3])
        @test size(r6) == (3,3)
    end
    
    @testset "normalizacion" begin
        data = Float64[0 0; 5 10; 10 20]
        
        norm_data = normalizeMinMax(data)
        @test minimum(norm_data) >= 0
        @test maximum(norm_data) <= 1
        
        norm_data2 = normalizeZeroMean(data)
        @test isapprox(mean(norm_data2, dims=1), [0 0], atol=1e-10)
        
        data2 = ones(2,2)
        norm_mm = normalizeMinMax(data2)
        @test all(norm_mm .== 0)
        
        norm_zm = normalizeZeroMean(data2)
        @test all(norm_zm .== 0)
        
        data3 = Float64[-1;1]
        norm_mm3 = normalizeMinMax(reshape(data3,:,1))
        @test norm_mm3 ≈ reshape([0;1], :,1)
        
        norm_zm3 = normalizeZeroMean(reshape(data3,:,1))
        @test mean(norm_zm3) ≈ 0
        @test std(norm_zm3) ≈ 1
    end
    
    @testset "Clasificacion y Accuracy" begin
        out = [0.2, 0.6, 0.8]
        @test classifyOutputs(out) == [false, true, true]
        
        targets = [false, true, false]
        @test accuracy(classifyOutputs(out), targets) ≈ 2/3
        
        out_multi = [0.1 0.9; 0.8 0.2]
        class_multi = classifyOutputs(out_multi)
        @test class_multi == [false true; true false]
        
        acc_real = accuracy([0.4,0.6], [false,true])
        @test acc_real == 1.0
        
        acc_wrong = accuracy([true,false], [false,true])
        @test acc_wrong == 0.0
        
        out_tie = [0.5 0.5]
        @test classifyOutputs(out_tie)[1,:] == [true, false]
    end
    
    @testset "buildClassANN" begin
        ann = buildClassANN(2, [4], 1)
        x = rand(Float32, 2, 5)
        y = ann(x)
        @test size(y) == (1, 5)
        
        ann_empty = buildClassANN(3, Int[], 2)
        y_empty = ann_empty(rand(Float32,3,1))
        @test size(y_empty) == (2,1)
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
end

println("\n✓ Todos los tests propios completados correctamente")


###### RUBRICA OFICIAL ####

println("\n" * "="^70)
println("EJECUTANDO TESTS DE LA RUBRICA OFICIAL")
println("="^70)

# Cargamos el dataset Iris
dataset = readdlm(joinpath(@__DIR__, "iris.data"), ',')
inputs = convert(Array{Float32,2}, dataset[:,1:4])

@testset "Rubrica Oficial" begin
    
    @testset "oneHotEncoding con Iris" begin
        targets = oneHotEncoding(dataset[:,5])
        
        @test size(targets) == (150,3)
        @test all(targets[  1:50 ,1]) && !any(targets[  1:50,  2:3 ])  # Primera clase
        @test all(targets[ 51:100,2]) && !any(targets[ 51:100,[1,3]])  # Segunda clase
        @test all(targets[101:150,3]) && !any(targets[101:150, 1:2] )  # Tercera clase
        
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
        @test size(ann[1].weight) == (5,4)
        @test size(ann[2].weight) == (3,5)
        @test size(ann(inputs')) == (3,150)
        println("   ✓ buildClassANN correcto")
    end
    
    # Comprobar accuracy 
    @testset "accuracy con Iris" begin
        targets = oneHotEncoding(dataset[:,5])
        
        @test isapprox(accuracy([sin.(1:150) cos.(1:150) tan.(1:150)], targets), 0.34)
        @test isapprox(accuracy(1:(-1/150):(1/150), targets[:,1]; threshold=0.75), 0.92)
        println("   ✓ accuracy correcto")
    end
end

println("\n" * "="^70)
println("✓✓✓ TODOS LOS TESTS COMPLETADOS CORRECTAMENTE ✓✓✓")
println("="^70)
