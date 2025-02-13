# Make Every Penny Count: Difficulty-Adaptive Self-Consistency for Cost-Efficient Reasoning [NAACL2025 Findings]

<div align="center">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Figure2.jpeg" width="700">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Figure3.jpeg" width="700">
</div>

**Self-consistency (SC), a widely used decoding strategy for chain-of-thought reasoning, shows significant gains across various multi-step reasoning tasks but comes with a high cost due to multiple 
sampling with the preset size. Its variants, ASC and ESC, dynamically adjust the number of samples based on the posterior distribution of a set of pre-samples, 
reducing the cost of SC with minimal impact on performance. Both methods, however, do not exploit the prior information about question difficulty. It often results in unnecessary repeated sampling for easy questions that could be accurately answered with just one attempt, wasting resources. 
To tackle this problem, we propose Difficulty-Adaptive Self-Consistency (DSC), which leverages the difficulty information from both prior and posterior perspectives to adaptively allocate inference resources, further reducing the overall cost of SC. 
To demonstrate DSC’s effectiveness, we conduct extensive experiments on three popular categories of reasoning tasks: arithmetic, commonsense and symbolic reasoning on six benchmarks. The empirical results show that DSC consistently surpasses the strong baseline ASC and ESC in terms of costs by a significant margin, while attaining comparable performances.**

More details for the use of code will be coming up soon.

## Results

<div align="center">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Table2.jpeg" width="700">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Table3.jpeg" width="700">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Table5.jpeg" width="700">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Table6.jpeg" width="700">
    <img src="https://github.com/WangXinglin/DSC/blob/main/Figure/Figure4.jpeg" width="700">
    
</div>


## Paper link
https://arxiv.org/html/2408.13457
