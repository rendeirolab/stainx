# Design components

The idea is to create a package that continues the scverse hackathon work
on gpu-accelerated stain normalization but with more optimizations and 
improved user experience.

The package is designed with the following components:

We will include several numerical methods for stain normalization:
1. Histogram matching
2. Reinhard
3. Mecenko
4. Vahadane

The implementations will include:
1. A pure pytorch-based implementation
2. A C++ implementation with optimized CUDA kernels

The user interface will only be available in python with:
- Skit-learn compatible API
- Torchvision/Albumentations transforms API

The state of the normalizer should be serializable. And all torch components should be scriptable.
(maybe saved as state dict? And provide a way to sync with huggingface?)

Example usage:

```python
# Automatically decide for the best implementation to use
from stainx import Reinhard
```

# Development plan

1. Core algorithms
2. API implementation
3. Extensive testing
4. Benchmarking against existing implementations
5. Documentation with examples

## Toolings

- UV for dependency management
- Prek for pre-commit hooks
- Pytest for testing
- mkdocs for documentation
- GitHub actions for CI
- PyPI for distribution

# Figure plan

### Proof of the importance of stain normalization in a histopathology task:
Construct a dataset for classification tasks,
where the train/val is specifically designed that each class is only scanned by one scanner type.
So during model training, the model will learn the scanner type rather than the class characteristics.
In a real-word held-out testing set, such a model will perform badly. We then apply the color normalization
during the training to account for the color differences. And the performance will be much better in the testing.
Also see (https://www.nature.com/articles/s41467-021-24698-1).

### Speed improvement:
Benchmarking against existing implementations. cuCIM, scikit-learn, torchstain, stainlib, etc.

What is the overhead when using stainx compared to the other implementations?

What is the GPU memory usage?
