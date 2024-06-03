using CUDA
using BenchmarkTools

const dim = 100_000_000
const a = 2.71828

# SAXPY on the CPU

x = ones(Float32, dim)
y = ones(Float32, dim)
z = zeros(Float32, dim)

println(typeof(x))

@btime z .= a .* x .+ y
cpu_time = 74.616

# SAXPY on the GPU

x = CUDA.ones(Float32, dim)
y = CUDA.ones(Float32, dim)
z = CUDA.zeros(Float32, dim)
println(typeof(x))

@btime CUDA.@sync z .= a .* x .+ y
gpu_time = 2.664
