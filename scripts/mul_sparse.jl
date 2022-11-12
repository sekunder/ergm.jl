using SparseArrays
using CUDA.CUSPARSE
using CUDA

n = 1000
p = 1e-3
A = convert(SparseMatrixCSC{Int}, sprand(Bool, n, n, p))
B = convert(SparseMatrixCSC{Int}, sprand(Bool, n, n, p))

function from_sparsearray(A)
    A_d = CuSparseMatrixCSR(A)
    matPtr = Ref{CUSPARSE.cusparseSpMatDescr_t}()
    CUSPARSE.cusparseCreateCsr(
        matPtr,
        n, n, A_d.nnz, 
        A_d.rowPtr, A_d.colVal, A_d.nzVal,
        CUSPARSE.CUSPARSE_INDEX_32I, CUSPARSE.CUSPARSE_INDEX_32I,
        CUSPARSE.CUSPARSE_INDEX_BASE_ZERO, CUDA.R_32F
    )
    mat = matPtr[]
    mat
end

opA = CUSPARSE.CUSPARSE_OPERATION_NON_TRANSPOSE
opB = CUSPARSE.CUSPARSE_OPERATION_NON_TRANSPOSE

matA = from_sparsearray(A)
matB = from_sparsearray(B)
matC = from_sparsearray(spzeros(n, n))

spgemmDescPtr = Ref{CUSPARSE.cusparseSpMatDescr_t}()
CUSPARSE.cusparseSpGEMM_createDescr(spgemmDescPtr)
spgemmDesc = spgemmDescPtr[]

handlePtr = Ref{CUSPARSE.cusparseHandle_t}()
CUSPARSE.cusparseCreate(handlePtr)
handle = handlePtr[]

alpha = Cfloat(1)
beta = Cfloat(0)
buffersize1Ptr = Ref{Csize_t}()
CUSPARSE.cusparseSpGEMM_workEstimation(
    handle, opA, opB, Ref(alpha), matA, matB, Ref(beta), matC,
    CUDA.R_32F, CUSPARSE.CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, buffersize1Ptr, CU_NULL
)
buffersize1 = buffersize1Ptr[]
dBuffer1 = Ptr{Cvoid}()
#CUDA.cudaMalloc(Ref(dBuffer1), buffersize1)
