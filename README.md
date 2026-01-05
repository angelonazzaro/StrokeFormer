# Introduction

Stroke is a common cerebrovascular disease and one of the leading causes of death in the
world. Accurate segmentation of stroke lesions is critical for diagnosis, treatment planning
and disease monitoring. Over the last decade, various segmentation approaches have been
proposed, with deep learning based methods becoming more and more essential for clinicians.
Despite their rapid growth, current methods are often highly specialized for a particular
imaging modality and stroke stage and result in heavy and complex pipelines that are
difficult to deploy in clinical settings.

In this work, we propose *StrokeFormer*, a _lightweight_ two-stage pipeline for stroke lesion
segmentation, designed for clinical deployment. StrokeFormer combines a 2D region proposal
mechanism with a 3D native segmentation model to reduce computational complexity,
mitigate severe class imbalance, and better exploit anatomical context.

# Objectives

Despite promising theoretical results, the practical effectiveness of the current pipeline is majorly limited
by the region proposal stage, which has to deal with _extreme_ class imbalance. In response, this work 
investigates and discusses strategies to combat overfitting and undesired biases 
in stroke lesion segmentation, with the goal of providing insights and guidance for future research efforts.


> [!WARNING]
> **Documentation Release Status:**
> The documentation is currently **under embargo**. The full source documentation will be released publicly in **August 2026**.
>
> This README serves as a general overview of the proposed method and documents its usage. 
> 

# StrokeFormer

The two phases of StrokeFormer are organized as follows:

- *Region Proposal*: Given a 3D volume, it is first
  decomposed into its 2D slices, then region proposal is performed independently on each. Since multiple
  regions can be proposed within a single slice, for example, when several lesions are present, we
  consolidate these into a single larger bounding box, that encompasses all proposed regions.   Because the RPN treats each slice independently, predicted regions may vary across consecutive
  slices, especially when anatomical perspectives vary between them. Therefore, for each 3D scan, we
  identify and group slices corresponding to distinct "anatomical regions" based on brain shape and
  size. To enforce spatial consistency, within each group, we enforce within each group the most frequent proposed region. If there is no most frequent region, a
  bigger bounding box is created to encompass all proposals. Region proposals are therefore aggregated within
  each group, and segmentation is performed independently for each anatomical region;


- *Segmentation*: The proposed regions are expanded to match a fixed RoI size (e.g. $64 \times 64
  \times 64$). When the 3D regions exceed this fixed resolution, we apply a sliding window approach
  over the larger region, processing overlapping sub-volume sequentially. The segmentation outputs
  from these overlapping windows are then averaged to produce the final prediction. Regions that
  were not proposed by the RPN model, and therefore not segmented, are masked to prevent the
  training signal from being distorted by non-relevant regions. The segmented outputs are finally
  concatenated back into a full 3D volume, reconstituting the whole scan.

# Data Augmentation

To mitigate class imbalance, we propose a data augmentation strategy to generate _anatomically plausible_ lesions. 
The strategy involves four main steps:

1. We identify similar healthy-sick pairs by comparing: brain size, shape and
  overall structure. Brain size and brain shape are compared using IoU, while overall structure
  similarity is computed through SSIM;

2. To ensure anatomical plausbility, we refine the previously selected pairs by retaining only
  those whose slice depth differs by no more than $30$ slices;

3. To induce and increase lesion variability, for each healthy slice, we randomly select lesions
  from its paired sick counterparts. Each lesion undergoes random roation and/or erosion;

4. Finally, the transformed lesions are transferred onto the healthy slice using Poisson image
  editing. To preserve the original slice intensities and avoid
  excessive blurring, we perform the transfer considering only the sick region on both slices.


# Installation Guide
To install the necessary requirements for the project, please follow the steps below.

## Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.11` or higher.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).
## Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 

## Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/angelonazzaro/StrokeFormer.git
```

## Installing Requirements

If you have an NVIDIA GPU with cuda installed on your system, run the following command in
your shell:

```shell
pip install -r requirements-cuda.txt
```

Otherwise, if your machine does not support cuda, run the following command in your shell to make
the code run on CPU:
```shell
pip install -r requirements.txt
```

