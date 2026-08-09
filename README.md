# 🔥 A novel power converters open-circuit fault diagnosis method with adaptive complex causal transformer

The implementation of the paper **[A novel power converters open-circuit fault diagnosis method with adaptive complex causal transformer](PAPER_LINK)**.

>  After publication, `PAPER_LINK` will be replaced with the official paper/DOI link.

## Updating!

[NEWS!] The code and documentation of **ACCT** are being continuously updated.

## Brief introduction

The phase and amplitude are two core elements in power converter monitoring signals. However, most existing diagnosis methods ignore phase information, resulting in poor diagnostic accuracy in some scenarios. To this end, a novel fault diagnosis framework named adaptive complex causal transformer (ACCT) is proposed to effectively achieve collaborative modeling of phase and amplitude information. Firstly, considering the concealment of phase feature extraction, a novel complex causal convolutional feature extractor is designed to decouple the amplitude and phase information from raw monitoring signals while adhering to temporal causality. Then, a causal spatial-temporal attention mechanism is developed to improve the fault diagnosis performance by capturing both temporal and spatial features. Subsequently, a dynamic trade-off parameter adjustment strategy is designed to adjust the optimization proportion between different losses, ensuring the objective achievement of each loss. Finally, compared with several well-known diagnosis methods, the experimental results on self-designed practical fault diagnosis hardware experimental platform demonstrate that ACCT possesses a more excellent diagnosis performance in both IID and Non-IID diagnosis tasks.


## Highlights

- A new **complex causal convolutional feature extractor** is designed to simultaneously decouple and extract phase & amplitude information from original real-valued monitoring signals. Meanwhile, a temporal causality layer is constructed to capture the temporal causal relationship. In addition, different from existing methods, a reconstruction loss is proposed to prevent semantic distortion between phase and amplitude information, thereby enhancing the representation capability for fault-related physical information.
  
- An innovative **causal spatial-temporal attention mechanism** is developed to parallelly capture and adaptively fuse temporal & spatial features while emphasizing the representation ability of each channel for the fault, further improving fault diagnosis capability of the model.
  
- A novel **dynamic trade-off parameter adjustment strategy** is built to coordinate the optimization proportions between different losses through feature clustering evaluation, thereby effectively achieving each optimization objective.

- This work designs and implements a self-designed practical fault diagnosis hardware experimental platform, which effectively verifies the diagnostic capability of the proposed **ACCT**.


## Paper

**A novel power converters open-circuit fault diagnosis method with adaptive complex causal transformer**

Yang Yu<sup>a</sup>, Quan Qian<sup>a,*</sup>, Fan Wu<sup>a</sup>, Chao He<sup>b</sup>, Kai Chen<sup>a</sup>, Yuhua Cheng<sup>a</sup>

<sup>a</sup> School of Automation Engineering, University of Electronic Science and Technology of China, Chengdu 611731, China

<sup>b</sup> State Key Laboratory of Advanced Rail Autonomous Operation, Beijing Jiaotong University, Beijing 100044, China

<sup>*</sup> Corresponding author: Quan Qian

**Paper link:** [To be updated](PAPER_LINK)

## The schematic diagram of the ACCT
<img width="4424" height="2179" alt="image67" src="https://github.com/user-attachments/assets/69e31e1b-2fdf-4fc7-b7bf-e7bca8a9f68c" />


## Citation

If you find this work useful in your research, please consider citing our paper.
