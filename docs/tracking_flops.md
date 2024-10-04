# Tracking Flops

Q1: The operations `sqrt`, `sin`, `sigmoid`, `log10`, `pow` take roughly the same time as scalar multiplication. Why?

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

Q: What is the arithmetic intensity for matrix multiplication and point-wise operations?

Q: When is the GPU memory bound?

Q: How do you explain the advertised FP32 performance (`0.32` TFLOPS/sec)?

Q: How can peak memory bandwidth be achieved?

---
## References

GPU SPECES:
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
MEMORY SPECS:
```
memory.total [MiB]
15360 MiB
15360 MiB
15360 MiB
15360 MiB
```