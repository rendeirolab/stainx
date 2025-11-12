# StainX Architecture Diagram

## Complete Project Architecture

To render the architecture diagram below, simply view this markdown file on GitHub or any markdown viewer that supports Mermaid diagrams. The following diagram visualizes the major layers and components within the StainX project, showing how user-facing classes, template abstractions, backend selection logic, and both PyTorch and CUDA implementations interact, along with supporting utilities.

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
 subgraph subGraph4["PyTorch Backend Implementations"]
        PTBB["PyTorchBackendBase"]
        HMPT["HistogramMatchingPyTorch"]
        REPT["ReinhardPyTorch"]
        MAPT["MacenkoPyTorch"]
        RGB2LAB["rgb_to_lab"]
        LAB2RGB["lab_to_rgb"]
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
        CFC["ChannelFormatConverter"]
  end
    NT -- inherits --> SNB
    SNB -- inherits --> nn.Module
    HM -- inherits --> NT
    RE -- inherits --> NT
    MA -- inherits --> NT
    NT -- selects backend via _select_backend --> PTBB
    NT -- selects backend via _select_backend --> CUBB
    HMPT -- inherits --> PTBB
    REPT -- inherits --> PTBB
    MAPT -- inherits --> PTBB
    PTBB -- provides static methods --> RGB2LAB & LAB2RGB
    HMCU -- inherits --> CUBB
    RECU -- inherits --> CUBB
    MACU -- inherits --> CUBB
    HMCU -- calls --> SC
    RECU -- calls --> SC
    MACU -- calls --> SC
    SC -- compiled from --> CU
    CU --> HM_CU & RE_CU & MA_CU
    UTILS --> GD & CFC
    REPT -- uses --> RGB2LAB & LAB2RGB
    SNB -- uses --> GD
    User -- creates --> HM & RE & MA
    style User fill:#e1f5ff
    style NT fill:#fff4e1
    style SNB fill:#fff4e1
    style PTBB fill:#e8f5e9
    style CUBB fill:#fce4ec
    style SC fill:#f3e5f5
```