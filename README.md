# Vulkan Radix Sort

**GPU Radix Sort** implemented in **Vulkan** and **GLSL**.

The implementation is based on Intel's Embree sorting kernel:

- https://github.com/embree/embree/blob/v4.0.0-ploc/kernels/rthwif/builder/gpu/sort.h

Tested on Linux with NVIDIA RTX 3070 GPU and on Linux with AMD Radeon RX Vega 7 GPU.

## (IMPORTANT) Single vs. Multi Radix Sort

This repository contains two implementations of the radix sort on the GPU: The `single_radixsort` uses only a single
work group, consists of one shader and is easy to set up. The `multi_radixsort` uses multiple work groups, consists of two
shaders that have to be executed several times in succession alternately and is therefore more complicated to set up but
yields higher performance for larger number of elements.

| `single_radixsort` (single work group)              | `multi_radixsort` (multiple work groups)                      |
|-----------------------------------------------------|---------------------------------------------------------------|
| + easy to set up                                    | - more complicated to set up                                  |
| + yields good performance for fewer elements (<10k) | + yields good performance even for a large number of elements |

For detailed information on integrating the shaders and for timings see below.

## (IMPORTANT) NVIDIA vs. AMD
The shaders are configured for NVIDIA GPUs. If you are using an AMD GPU, set `SUBGROUP_SIZE=64` in [single_radixsort.comp](https://github.com/MircoWerner/VkRadixSort/blob/main/singleradixsort/resources/shaders/single_radixsort.comp#L12) or [multi_radixsort.comp](https://github.com/MircoWerner/VkRadixSort/blob/main/multiradixsort/resources/shaders/multi_radixsort.comp#L13).

## Table of Contents

- [Example Usage](#example-usage) (reference implementation in Vulkan)
    - [Compile / Run](#compile--run)
    - [Interesting Files](#interesting-files)
- [Own Usage: Single Radix Sort](#single--own--usage) (how to use the `single_radixsort` / the compute shader in your
  own Vulkan project)
    - [Shaders / Compute Pass](#single--shaders--compute-pass)
    - [Buffers](#single--buffers)
    - [Push Constants](#single--push--constants)
    - [Execute](#single--execute)
- [Own Usage: Multi Radix Sort](#multi--own-usage) (how to use the `multi_radixsort` / the compute shaders in your own
  Vulkan project)
    - [Number of Blocks per Work Group](#multi--numblocks)
    - [Shaders / Compute Pass](#multi--shaders--compute-pass)
    - [Buffers](#multi--buffers)
    - [Push Constants](#multi--push--constants)
    - [Execute](#multi--execute)
- [Timings](#timings)

<a name="example--usage"></a>

## Example Usage

This repository contains a reference implementation to show how to use the provided shaders.

<a name="compile--run"></a>

### Compile / Run

Requirements: Vulkan, glm

```bash 
git clone --recursive git@github.com:MircoWerner/VkRadixSort.git
cd VkRadixSort
mkdir build
cd build
cmake ..
make
```

`single_radixsort`

```bash
cd singleradixsort
./singleradixsortexample
```

`multi_radixsort`

```bash
cd multiradixsort
./multiradixsortexample
```

<a name="interesting--files"></a>

### Interesting Files

|                                                                                | `single_radixsort`                                                                                 | `multi_radixsort`                                                                                                               |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| shaders                                                                        | `singleradixsort/resources/shaders/single_radixsort.comp`                                          | `multiradixsort/resources/shaders/multi_radixsort_histograms.comp` <br> `multiradixsort/resources/shaders/multi_radixsort.comp` |
| compute pass                                                                   | `singleradixsort/include/SingleRadixSortPass.h` <br> `singleradixsort/src/SingleRadixSortPass.cpp` | `multiradixsort/include/MultiRadixSortPass.h` <br> `multiradixsort/src/MultiRadixSortPass.cpp`                                  |
| program logic (buffer definition,<br />assigning push constants, execution...) | `singleradixsort/include/SingleRadixSort.h` <br> `singleradixsort/src/SingleRadixSort.cpp`         | `multiradixsort/include/MultiRadixSort.h` <br> `multiradixsort/src/MultiRadixSort.cpp`                                          |

<a name="single--own--usage"></a>

## Own Usage: Single Radix Sort

Explanation how to use the `single_radixsort` in your own Vulkan project.
Assume you have a vector/array of `uint32_t` (you have to preprocess negative numbers; uint64_t requires to adjust the
number of iterations defined in the shader) with a size of `NUM_ELEMENTS`.

<a name="single--shaders--compute-pass"></a>

### Shaders / Compute Pass

Copy the following [shader](https://github.com/MircoWerner/VkRadixSort/tree/main/singleradixsort/resources/shaders) to
your project:

```
single_radixsort.comp
```

Create a compute pass consisting of the single compute shader.

Set the global invocation size of the shader:

```cpp
single_radixsort: (256, 1, 1) // 256=WORKGROUP_SIZE defined in single_radixsort.comp, i.e. we just want to launch a single work group
```

<a name="single--buffers"></a>

### Buffers

Create the following two buffers and assign them to the following sets and indices of your compute pass:

| buffer    | size (bytes)                    | initialize         | (set,index) |
|-----------|---------------------------------|--------------------|-------------|
| m_buffer0 | NUM_ELEMENTS * sizeof(uint32_t) | vector of elements | (0,0)       |
| m_buffer1 | NUM_ELEMENTS * sizeof(uint32_t) | -                  | (0,1)       |

Use `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` and `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.
`m_buffer0` is the input buffer and will also contain the sorted output. `m_buffer1` is used during the computation (ping pong buffer).

<a name="single--push--constants"></a>
### Push Constants

Define the following push constant struct for the shader and set its data:

```cpp
struct PushConstants {
    uint32_t g_num_elements; // = NUM_ELEMENTS
};
```

<a name="single--execute"></a>
### Execute

Execute the compute pass. Wait for the compute queue to idle. The result is in the `m_buffer0` buffer.

<a name="multi--own-usage"></a>
## Own Usage: Multi Radix Sort

Explanation how to use the `multi_radixsort` in your own Vulkan project.
Assume you have a vector/array of `uint32_t` (you have to preprocess negative numbers; uint64_t requires to adjust the
number of iterations defined in the shader) with a size of `NUM_ELEMENTS`.

<a name="multi--numblocks"></a>
### Number of Blocks per Work Group

Each thread in a work group normally processes a single element. This is slow because a lot of work groups have to be
launched. Instead, the number of elements each thread of a work group processes ("number of processed blocks per work
group") can be increased. Assume this is stored in `NUM_BLOCKS_PER_WORKGROUP` (uint32_t, >= 1). 
See [Timings](#timings) for more information on how to choose the parameter.

<a name="multi--shaders--compute-pass"></a>

### Shaders / Compute Pass

Copy the following [shaders](https://github.com/MircoWerner/VkRadixSort/tree/main/multiradixsort/resources/shaders) to
your project:

```
multi_radixsort_histograms.comp
multi_radixsort.comp
```

Create a compute pass consisting of the two compute shaders (in the order shown above) with pipeline barriers after each
of them:

```
VkMemoryBarrier memoryBarrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
```

Set the global invocation sizes of the two shaders:

```cpp
uint32_t globalInvocationSize = NUM_ELEMENTS / NUM_BLOCKS_PER_WORKGROUP;
uint32_t remainder = NUM_ELEMENTS % NUM_BLOCKS_PER_WORKGROUP;
globalInvocationSize += remainder > 0 ? 1 : 0;

multi_radixsort_histograms: (globalInvocationSize, 1, 1) // (x,y,z global invocation sizes)
multi_radixsort: (globalInvocationSize, 1, 1)
```

We will later execute this pass (consisting of the two successive shaders) four times to first sort the lower 8 bits,
then the next higher 8 bits...

<a name="multi--buffers"></a>
### Buffers

Create the following three buffers and assign them to the following sets and indices of your compute pass.
Since we use the `m_buffer0` and `m_buffer1` alternating as input/output in the four iterations (ping pong buffers), the
buffers have to be bound to different indices in different iterations.

| buffer            | size (bytes)                                              | initialize         | (set,index)                                                      |
|-------------------|-----------------------------------------------------------|--------------------|------------------------------------------------------------------|
| m_buffer0         | NUM_ELEMENTS * sizeof(uint32_t)                           | vector of elements | iterations 0 and 2: (0,0),(1,0) <br/> iterations 1 and 3: (1,1)  |
| m_buffer1         | NUM_ELEMENTS * sizeof(uint32_t)                           | -                  | iterations 0 and 2: (1,1) <br/> iterations 1 and 3: (0,0), (1,0) |
| m_bufferHistogram | NUMBER_OF_WORKGROUPS * RADIX_SORT_BINS * sizeof(uint32_t) | -                  | (0,1),(1,2)                                                      |

- `NUMBER_OF_WORKGROUPS`: number of work groups / dispatch size (depends on the global invocation size) `(globalInvocationSize + workGroupSize - 1) / workGroupSize` (`workgroupSize=256` defined in the shader)
- `RADIX_SORT_BINS=256`: we sort 8 bits in each iteration, i.e. 2^8=256

Use `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` and `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

<a name="multi--push--constants"></a>
### Push Constants

Define the following push constant structs for the two shaders and set their data:

```cpp
struct PushConstantsHistograms {
    uint32_t g_num_elements; // == NUM_ELEMENTS
    uint32_t g_shift; // (*)
    uint32_t g_num_workgroups; // == NUMBER_OF_WORKGROUPS as defined in the section above
    uint32_t g_num_blocks_per_workgroup; // == NUM_BLOCKS_PER_WORKGROUP
};

struct PushConstants {
    uint32_t g_num_elements; // == NUM_ELEMENTS
    uint32_t g_shift; // (*)
    uint32_t g_num_workgroups; // == NUMBER_OF_WORKGROUPS as defined in the section above
    uint32_t g_num_blocks_per_workgroup; // == NUM_BLOCKS_PER_WORKGROUP
};
```

(*) The shift has to be set to `0` in iteration zero and to `8`, `16`, `24` in iteration one, two, three respectively.

<a name="multi--execute"></a>
### Execute
Execute the compute pass four times (remember to adjust the buffer bindings and shifts in each iteration). Wait for the compute queue to idle. The result is in the `m_buffer0` buffer.

<a name="timings"></a>
## Timings
Tests performed on NVIDIA GeForce RTX 3070 8GB and AMD Ryzen 5 2600 with 2x Crucial RAM 16GB DDR4 3200MHz.
`NUM_ELEMENTS` random `uint32_t` are sorted (generated with [mt19937](https://cplusplus.com/reference/random/mt19937/), see [here in the code](https://github.com/MircoWerner/VkRadixSort/blob/main/multiradixsort/src/MultiRadixSort.cpp#L117)).
**Notice that these are log scale plots!**
When sorting only a small number of elements, the runtime difference between the two GPU sorting variants is smaller.
With an increase in the number of elements to sort, the advantage of using multiple workgroups becomes clear.

![img comparison](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_comparison.png?raw=true)

### NUM_BLOCKS_PER_WORKGROUP
Now let us take a look at the `NUM_BLOCKS_PER_WORKGROUP` parameter in the `multi_radixsort`.
As already mentioned in the section [Number of Blocks per Work Group](#multi--numblocks), it is slow if each thread in a work group processes only a single element.
For 1,000,000 elements, the `single_radixsort` with only one work group takes 18.973ms.
Setting `NUM_BLOCKS_PER_WORKGROUP=1` takes 30.04ms, since a lot of work groups (3907 = ceil(1,000,000/256/1)) have to be executed (high scheduling overhead).
On the other hand, setting `NUM_BLOCKS_PER_WORKGROUP=4096` takes 20.91ms and uses only a single work group (1 = ceil(1,000,000/256/4096)).
As we can see, setting `NUM_BLOCKS_PER_WORKGROUP=4096` is the same as using the `single_radixsort` (and requires approximately the same time).
For a certain number of elements, we can find the best tradeoff between scheduling overhead and using too few threads.
For 1,000,000 elements, this sweet spot is for my GPU `NUM_BLOCKS_PER_WORKGROUP=32` (using 123 work groups).
I have done this optimization for the other tested number of elements (100, 1000, ...).
You can see the results in the diagrams in the spoiler below.
The optimized results are shown in the diagram above for the `multi_radixsort` line.

![img 1000000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_1000000.png?raw=true)

<details>
  <summary>Timings for NUM_BLOCKS_PER_WORKGROUP for 100-100,000,000 elements</summary>

![img 100](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_100.png?raw=true)

![img 1000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_1000.png?raw=true)

![img 10000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_10000.png?raw=true)

![img 100000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_100000.png?raw=true)

![img 1000000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_1000000.png?raw=true)

![img 10000000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_10000000.png?raw=true)

![img 100000000](https://github.com/MircoWerner/VkRadixSort/blob/main/timings/radixsort_multi_100000000.png?raw=true)

</details>