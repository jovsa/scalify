# Tracking Flops

### Q: The operations `sqrt`, `sin`, `sigmoid`, `log10`, `pow` take roughly the same time as scalar multiplication. Why?

[link to code](https://github.com/jovsa/scalify/blob/main/scalify/flops.py)

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                  cudaDeviceSynchronize        99.02%     250.445ms        99.02%     250.445ms     250.445ms       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b             1
                                             aten::ones         0.01%      22.216us         0.84%       2.117ms       2.117ms       0.000us         0.00%       1.141ms       1.141ms           0 b           0 b     256.00 Mb           0 b             1
                                            aten::empty         0.81%       2.040ms         0.81%       2.040ms       2.040ms       0.000us         0.00%       0.000us       0.000us           0 b           0 b     256.00 Mb     256.00 Mb             1
                                       cudaLaunchKernel         0.04%     113.258us         0.04%     113.258us      10.296us       0.000us         0.00%       0.000us       0.000us           0 b           0 b           0 b           0 b            11
                                           aten::matmul         0.00%       5.966us         0.04%      95.100us      95.100us       0.000us         0.00%     230.208ms     230.208ms           0 b           0 b           0 b           0 b             1
                                               aten::mm         0.03%      72.594us         0.04%      89.134us      89.134us     230.208ms        91.71%     230.208ms     230.208ms           0 b           0 b           0 b           0 b             1
                                              aten::sum         0.02%      41.065us         0.03%      65.381us      65.381us     991.583us         0.40%     991.583us     991.583us           0 b           0 b         512 b         512 b             1
                                            aten::fill_         0.01%      24.671us         0.02%      54.570us      54.570us       1.141ms         0.45%       1.141ms       1.141ms           0 b           0 b           0 b           0 b             1
                                              aten::pow         0.01%      26.403us         0.01%      35.959us      35.959us       2.167ms         0.86%       2.167ms       2.167ms           0 b           0 b     256.00 Mb     256.00 Mb             1
                                             aten::mul_         0.01%      26.698us         0.01%      35.701us      35.701us       2.184ms         0.87%       2.184ms       2.184ms           0 b           0 b           0 b           0 b             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 252.927ms
Self CUDA time total: 251.008ms
```

In pytorch, the main types of [operators](https://dev-discuss.pytorch.org/t/where-do-the-2000-pytorch-operators-come-from-more-than-you-wanted-to-know/373) affected here are composite pointwise/reductions.

Before the GPU can perform numerical operations on data, that data has to first be read into registers.

If peak bandwith of T4 GPU is advertised as `0.32` TB/s, this means a sqaure matrix of size `8192` matrix of `fp32` values takes at least $$(4 * (8192)^2)/(0.32*10**12)) = 0.839 ms $$  ms to read. Hence:
* `0.839*2`= `1.676` ms (assuming read + write round trip)
* Therefore, `0.040` TFLOPS/sec ~ `12%`

### Q: What is the arithmetic intensity for matrix multiplication and point-wise operations?

`Arithmetic intensity` = `total flop count`/ `read/write bytes`

Hence,
For matrix multiplication of two `n` x `n` matrices, the flop count is  `O(n^3)`,
when planned carefully the number of bytes read/written are `O(n^2)`
so the arithmetic intensity is `O(n)`

For point-wise operations, the flop count is
, where
 is the number of tensor entries; the bytes read/written is also
 so the arithmetic intensity is
.

### Q: When is an A100 (40 GB) memory bandwidth bound?**

An A100 (40 GB) can perform roughly
 flops in the time it takes to read or write
 bytes (single precision float). If the arithmetic intensity of a program <
 on an A100 (40 GB) then it will be bound by memory bandwidth.

### Q: How do you explain the advertised FP32 performance (19.5 TFLOPS/sec) from the A100 microarchitecture?**

The maximum clock speed of an A100 is `1410` MHz. Dividing the advertised flops by the clock speed, we get
 FLOPS/cycle. Looking at the spec sheet, we see it has `6912` “CUDA Cores” - this number is suspiciously close to `13830/2` = `6915`
.

A CUDA Core is essentially one single precision floating point unit. Multiplying two n-bit numbers entails adding n-1 partial products. Therefore a MAC (multiply-and-accumulate) operation has very little incremental cost over multiplication. All hardware vendors consider a MAC to be two flops - so `6912` cores can perform ` 13824` FLOPS/cycle. The discrepancy with above is due to rounding.

Even for compute-bound workloads achieving peak flops is hard - every CUDA Core has to perform a MAC every single cycle.

### Q: How can peak memory bandwidth be achieved?

Just like TFLOPS/sec, the memory bandwidth on GPU spec sheets can easily be misunderstood. It corresponds to best case memory layouts i.e., when copies are aligned with cache dimensions. The high memory bandwidth is due to a very wide memory bus (`5120`bits) which is roughly  that of a CPU. It is optimal for reading/writing to/from long contiguous segments of memory.

While reading/writing tensors are an ideal use case some challenges still remain. E.g., consider the case of transposing a 2D matrix - if the matrix is read in column major order then a naive approach to writing the result will be fragmented. The solution is to use clever memory access by [coalescing reads](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/) and using [shared memory](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/).


---
## References

`$lspci`

**GPU SPECES USED:**
```
00:00.0 Host bridge: Intel Corporation 440FX - 82441FX PMC [Natoma] (rev 02)
00:01.0 ISA bridge: Intel Corporation 82371AB/EB/MB PIIX4 ISA (rev 03)
00:01.3 Bridge: Intel Corporation 82371AB/EB/MB PIIX4 ACPI (rev 03)
00:03.0 Non-VGA unclassified device: Red Hat, Inc. Virtio SCSI
00:04.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:05.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:06.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:07.0 3D controller: NVIDIA Corporation TU104GL [Tesla T4] (rev a1)
00:08.0 Ethernet controller: Red Hat, Inc. Virtio network device
00:09.0 Unclassified device [00ff]: Red Hat, Inc. Virtio RNG
```

`$nvidia-smi --query-gpu=memory.total --format=csv`

**MEMORY SPECS USED:**
```
memory.total [MiB]
15360 MiB
15360 MiB
15360 MiB
15360 MiB
```