NVLink’s low latency and high throughput allow GPUs to act as one large memory pool for model training.

NVSwitch hubs inside H100 systems provide 64 NVLink ports each, and new external NVLink Switch chips extend this all-to-all fabric across multiple servers​
DEVELOPER.NVIDIA.COM
. In fact, up to 32 H100 nodes (256 GPUs) can be fully interconnected via NVLink in a fat-tree topology, delivering an enormous 57.6 TB/s of all-to-all bandwidth within that cluster segment​
DEVELOPER.NVIDIA.COM
. This capability, used in NVIDIA’s DGX SuperPOD designs, means a group of 256 GPUs can communicate as if on one giant 256-way NVSwitch – a critical feature for large models that require frequent internode communication.

Google’s TPU-based systems use their proprietary interconnect instead of InfiniBand. TPU v3/v4 pods arrange accelerators in a 2D toroidal mesh network, with optical links between chips. A TPU v4 pod (4096 chips) doesn’t have full all-to-all wiring; instead each chip communicates with a few neighbors at a time. Google’s software stack (JAX/Pathways) implements collectives over this mesh. For instance, to train PaLM 540B, Google connected two TPU v4 pods (6144 chips total) and managed communication across pods using the Pathways system​
RESEARCH.GOOGLE
. While individual TPU links are slower than NVLink, the massive scale (6K+ accelerators) and Google’s efficient scheduling still yielded good utilization (PaLM reached 57.8% of hardware FLOPS utilization at that scale by overlapping communication with computation)​

This unprecedented scale highlights how far networking technology has come – a single training run can now span 10^5 accelerators, provided the network is fast enough to keep them fed with data.


Microsoft’s DeepSpeed and NVIDIA’s Megatron-LM both support pipelining; the 530B MT-NLG model was trained with 8-way pipeline parallelism (8 layers-deep pipeline) across nodes​
RESEARCH.GOOGLE
. Pipeline parallelism introduces “bubble” overhead (under-utilized time while the pipeline fills and drains) and complicates scheduling, but it reduces memory pressure by not replicating the entire model on every GPU. Companies try to minimize pipeline stages (since it’s less flexible for dynamic workloads) – OpenAI’s later models favored large all-reduce domains (data parallel) and tensor-slicing, using pipeline parallelism only if absolutely required by memory constraints.

Microsoft’s DeepSpeed-MoE combined data, model, and expert parallelism to train a 3.5 trillion parameter MoE on 512 GPUs, achieving near-linear scaling and ~100 TFLOPS per GPU​
MICROSOFT.COM
. The key to expert parallelism is an all-to-all communication pattern: tokens must be scattered to the GPU hosting their chosen expert and then results gathered back. DeepSpeed’s system optimized this by grouping tokens by expert and performing all-to-all transfers within subgroups to limit network traffic​
PROCEEDINGS.MLR.PRESS
. One notable MoE success story is DeepSeek (a Chinese startup’s open-source LLM): their flagship model DeepSeek-R1 uses a 64-expert mixture-of-experts architecture with 671 billion parameters​
TECHTARGET.COM
. Thanks to MoE, only ~37B parameters are active per query (about 5% of the model)​
DAILY.DEV
, drastically cutting the compute and memory required for each token. DeepSeek leveraged expert parallelism to train this model on relatively fewer GPUs and at lower cost – under $6M according to the company​
TECHTARGET.COM
 – a disruptive achievement given that dense models like GPT-4 are estimated to cost tens of millions. In sum, expert parallelism has emerged as an attractive strategy to push model scale without a proportional increase in training cost, but it hinges on extremely efficient all-to-all communication and software that can orchestrate sparse activation.

 In contrast, NVIDIA/Microsoft’s Megatron-Turing NLG 530B (2021) used all three: tensor (model) slicing + pipeline stages (to distribute the network across 280 nodes) + data parallel across multiple such pipelines​
RESEARCH.GOOGLE
. OpenAI’s GPT-4 is believed to have used a similar Megatron approach on Azure’s GPU cluster – many experts suspect GPT-4 runs on 8-way model parallel within each node (so the ~1 trillion parameters model is partitioned across 8 GPUs or more) combined with thousands of nodes in data parallel. 

 Meanwhile, Meta’s LLaMA models (2023) took a slightly different route, leveraging FSDP (Fully Sharded Data Parallel) which is a variant of model+data parallelism: it partitions each layer’s weights across GPUs and only keeps a single shard of the optimizer state per GPU, reducing memory redundancy. LLaMA-65B was trained on 2,048 A100 GPUs primarily with sharded data parallelism (no MoE, no pipeline needed at that scale), making it relatively straightforward to train compared to earlier pipelines. All these configurations underscore a common theme: as model sizes exploded, organizations innovated hybrid parallelism schemes to balance memory usage, compute load, and communication overhead.

 For example, on a 64-GPU cluster, enabling NVIDIA’s in-network reduction (SHARP) can drive all-reduce throughput to 227 GB/s (aggregate)​
HIBD.CSE.OHIO-STATE.EDU
 – nearly saturating a 400 Gb/s network across 8 nodes.

 All-to-All and Scatter/Gather: Unlike all-reduce (which aggregates identical shapes on all ranks), all-to-all involves each GPU exchanging different slices of data with every other GPU. This pattern arises in Mixture-of-Experts training and in some model-parallel schemes

 For MoE, after a routing decision, each GPU may hold tokens that belong to many different experts; an all-to-all shuffle redistributes tokens so that each expert GPU receives just the tokens intended for it. This can be a heavier communication load than all-reduce because the data volume scales with the batch size and hidden dimensionality. For instance, DeepSeek’s 671B MoE model uses 64 experts – effectively 64-way all-to-all traffic at certain layers. Each token’s embedding (e.g. a vector of length 12k or more, which is ~24 KB in FP16) must be sent to the appropriate expert. If a mini-batch has, say, 2048 tokens spread across experts, the all-to-all could involve tens of MB moving between every pair of GPUs

 Efficient implementation is crucial: Microsoft’s DeepSpeed MoE groups tokens by destination expert and overlaps communication with computation to maximize throughput​
PROCEEDINGS.MLR.PRESS
. Research prototypes like Tutel and FastMoE similarly optimize the scatter/gather of activations for MoE. Another case of scatter/gather is pipeline parallelism: micro-batches of activations are sent from one pipeline stage to the next (scatter) and eventually one collects outputs from the final stage (gather). This is typically one-directional and easier to overlap, but still relies on low latency links to avoid stalling the pipeline.

Packet Size and Bandwidth Utilization: LLM training exhibits a mix of communication sizes. Gradients of full layers are large (megabytes), which favors high-bandwidth protocols. NCCL uses a “Simple” protocol for large messages that achieves maximal bandwidth by sending big chunks (e.g. 256KB or 1MB) in a pipelined fashion​
PARSA.EPFL.CH
​
PARSA.EPFL.CH
. Small messages (like syncing small layers or attention masks) use a lower-latency algorithm (NCCL’s LL or LL128 protocol) to minimize overhead​
PARSA.EPFL.CH
. Engineers tune these thresholds to the network’s sweet spot. On modern systems, once message sizes exceed a few MB, the network links (NVLink or NIC) can be kept nearly full. 

In summary, the communication backbone for LLM training has evolved to be collective-aware and topology-aware. All-reduce ops are offloaded and hierarchically organized, all-to-all ops are carefully scheduled to avoid network hot spots, and every ounce of bandwidth (NVLink’s hundreds of GB/s and InfiniBand’s tens of GB/s) is utilized via pipelining. These advances in the communication stack – from NCCL/MPI improvements to smart networking hardware – are what allow model parallelism and data parallelism to scale so effectively in today’s datacenter environments.

 In practice, this means layers like matrix multiplies in the transformer block can run in FP8, while more sensitive operations (layer norm, softmax, residual additions) stay in FP16/BF16.

 Techniques like GPTQ and INT8 quantization were popular in 2022 for LLM inference. Hardware like NVIDIA’s Hopper and Ada GPUs and Google’s TPU v5e offer dedicated support for 8-bit matrix multiplies for inference​
RCRWIRELESS.COM
. The difference with FP8 for training is handling the dynamic range and precision on the fly during backpropagation. So far, results are promising – models like GPT-3, PaLM, and vision transformers have been trained to high accuracy with FP8 in research trials​
DEVELOPER.NVIDIA.COM

 Notably, Meta’s latest GPUs in Grand Teton systems not only use FP8 for compute, but also support an FP8 format cache for the attention key-values to handle extremely long context windows efficiently​
THEREGISTER.COM
. 

Each input only triggers 3–5 experts (about 37B parameters worth)​
DAILY.DEV
, meaning the effective model capacity is huge, but the compute per token is much lower than an equivalent dense model. This design, combined with large-scale reinforcement learning on reasoning tasks and aggressive distillation, allowed DeepSeek to achieve GPT-4-level performance at a tiny fraction of the compute budget​
TECHTARGET.COM
​
TECHTARGET.COM
. Notably, DeepSeek did this without access to top-of-the-line hardware: due to export restrictions, they likely trained on NVIDIA A800 GPUs (slightly nerfed A100s)​
TECHTARGET.COM
 and possibly used only hundreds of GPUs rather than thousands

 