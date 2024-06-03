using BenchmarkTools, CUDA

const dim = 100_000_000
const a = 2.71828

cpu_time = 71 # approx from saxpy.jl
broadcast_time = 2.6 # approx from saxpy.jl

# SAXPY using CUDA kernel
x = CUDA.ones(Float32, dim)
y = CUDA.ones(Float32, dim)
z = CUDA.zeros(Float32, dim)

# calculate threads per block and blocks per grid
nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
)

nblocks = cld(dim, nthreads) # cld means ceiling division

function saxpy_gpu_kernel!(z, a, x, y)
    # calculate index number
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # calculate SAXPY
    if i <= length(z)
        @inbounds z[i] = a * x[i] + y[i]
    end
    return nothing
end

@btime CUDA.@sync @cuda(
    threads=nthreads,
    blocks=nblocks,
    saxpy_gpu_kernel!(z, a, x, y)
)

# gives the following result
kernel_time = 2.709

# the tutorial also compares using Julia wrappers
# for CUBLAS library functions
# I skipped this because those are apparently being phased out
