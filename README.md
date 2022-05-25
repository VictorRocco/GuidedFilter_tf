# Guided Filter (GF) based layers for TensorFlow
The [Guided Filter](https://en.wikipedia.org/wiki/Guided_filter) is a technique for edge-aware image filtering.

Is one of several popular algorithms for edge-preserving smoothing, like [Bilateral Filter](https://en.wikipedia.org/wiki/Bilateral_filter).

GF helps to improve the performance of multiple computer vision tasks, including faster training of [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) models.

Here we implement the original Guided Filter for Gray and RGB images. 
The variant Fast Guided Filter (FGF) is included as an optional parameter.

In the future, we will implement or port other variants: 
- [Fast End-to-End Trainable Guided Filter](https://arxiv.org/abs/1803.05619)
- [Unsharp Mask Guided Filtering](https://arxiv.org/abs/2106.01428)
- [Robust Guided Image Filtering](https://arxiv.org/pdf/1703.09379.pdf)
- And more...

## Properties
#### [Edge-preserving filtering](https://en.wikipedia.org/wiki/Edge-preserving_smoothing)

When the guidance image (I) is used at the same time as the filtering image (p), the guided filter removes noise while preserving clear edges.

Specifically, a “flat patch” or a “high variance patch” can be specified by the parameter (ϵ).
Patches with variance much lower than the parameter (ϵ) will be smoothed, and those with variances much higher than (ϵ) will be preserved. 

#### Gradient-preserving filtering

The guided filter performs better in avoiding gradient reversal, moreover, in some cases, it can be ensured that it does not occur.

When using the bilateral filter, artifacts may appear on the edges. This is because of the pixel value's abrupt change on the edge. 

#### Structure-transferring filtering

Due to the local linear model of Guided Filter, it is possible to transfer the structure from the guidance image (I)  to the output image (q). 

This property enables some special filtering-based applications, such as feathering, matting and dehazing. 

## Implementation on TensorFlow 2

At the time of implementation there wasn't a full implementation on Tensorflow 2.0 (Gray and RGB Images, Guided Filter and Fast variant). So I had to merge from different sources and port from different languages while reading from the corresponding papers and presentation slides.

- Based on the paper [Guided Image Filtering](http://kaiminghe.com/publications/eccv10guidedfilter.pdf)
- Based on the paper [Fast Guided Filter](https://arxiv.org/abs/1505.00996).
- Based on the presentation slides [Guided Image Filtering](http://kaiminghe.com/eccv10/eccv10ppt.pdf)
- Based on the implementation of [Huikai Wu, Shuai Zheng, Junge Zhang and Kaiqi Huang](https://github.com/wuhuikai/DeepGuidedFilter/tree/master/GuidedFilteringLayer/GuidedFilter_TF). Only for Gray Images, ported from Tensorflow 1.0.
- Based on the implementation of [Fast Guided Filter by Kaiming He](https://github.com/accessify/fast-guided-filter). Ported from Matlab.
- Based on the implementation of [Guided Filter](https://github.com/lisabug/guided-filter/blob/master/core/filter.py). Ported from Python/Numpy.
## Install
```
make clean install
```

## Example usage
```
from GuidedFilter.GuidedFilter import GuidedFilter
    
GF = GuidedFilter(radious=1, eps=1.0, nhwc=True)
fgf_output = GF(input, input)

``` 

## Example explanation
radious is the radious to apply the filter. 
Think of it as the spatial width, the distance of pixels in 2D.

eps stands for "Edge Preserve Smooth". 
Think of it as the spatial depth, the difference of color between adjacent pixels.

nhwc stands for "channel last format". N (batch size) Height Width Channels. 
Default to true because it is the standard in Tensorflow.

