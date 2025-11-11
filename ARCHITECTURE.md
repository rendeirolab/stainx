# StainX Architecture Diagram

## Complete Project Architecture

To render the architecture diagram below, simply view this markdown file on GitHub or any markdown viewer that supports Mermaid diagrams. The following diagram visualizes the major layers and components within the StainX project, showing how user-facing classes, template abstractions, backend selectors, and both PyTorch and CUDA implementations interact, along with supporting utilities.

```mermaid
---
config:
  layout: elk
---
flowchart TB
 subgraph subGraph0["User API Layer"]
        User["User Code"]
        HM["HistogramMatching"]
        RE["Reinhard"]
        MA["Macenko"]
  end
 subgraph subGraph1["Normalizer Template Layer"]
        NT["NormalizerTemplate"]
        SNB["StainNormalizerBase"]
        nn.Module["torch.nn.Module"]
  end
 subgraph subGraph3["Backend Selection"]
        BS["Backend Selector"]
        PTB["PyTorch Backend"]
        CUB["CUDA Backend"]
  end
 subgraph subGraph4["PyTorch Backend Implementations"]
        PTBB["PyTorchBackendBase"]
        HMPT["HistogramMatchingPyTorch"]
        REPT["ReinhardPyTorch"]
        MAPT["MacenkoPyTorch"]
  end
 subgraph subGraph5["CUDA Backend Implementations"]
        CUBB["CUDABackendBase"]
        HMCU["HistogramMatchingCUDA"]
        RECU["ReinhardCUDA"]
        MACU["MacenkoCUDA"]
  end
 subgraph subGraph6["CUDA Extension"]
        SC["stainx_cuda"]
        CU[".cu files"]
        HM_CU["histogram_matching.cu"]
        RE_CU["reinhard.cu"]
        MA_CU["macenko.cu"]
  end
 subgraph Utilities["Utilities"]
        UTILS["utils.py"]
        GD["get_device"]
        RGB2LAB["rgb_to_lab"]
        LAB2RGB["lab_to_rgb"]
  end
    NT -- inherits --> SNB
    SNB -- inherits --> nn.Module
    HM -- inherits --> NT
    RE -- inherits --> NT
    MA -- inherits --> NT
    NT -- selects --> BS
    BS -- chooses --> PTB & CUB
    PTB --> PTBB
    PTBB --> HMPT & REPT & MAPT
    CUB --> CUBB
    CUBB --> HMCU & RECU & MACU
    HMCU -- calls --> SC
    RECU -- calls --> SC
    MACU -- calls --> SC
    SC -- compiled from --> CU
    CU --> HM_CU & RE_CU & MA_CU
    UTILS --> GD & RGB2LAB & LAB2RGB
    REPT -- uses --> RGB2LAB & LAB2RGB
    User -- creates --> HM & RE & MA
    style User fill:#e1f5ff
    style NT fill:#fff4e1
    style SNB fill:#fff4e1
    style PTB fill:#e8f5e9
    style CUB fill:#fce4ec
    style SC fill:#f3e5f5
```