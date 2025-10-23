# Exposed Parameters - Summary

## New Parameters Added to TripoSR Gradio Interface

### ✅ Successfully Exposed Parameters

All parameters have been exposed with default values matching the original hardcoded settings.

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| **Marching Cubes Resolution** | 320 | 32-512 | Quality vs Speed - Higher = more detail but slower |
| **Chunk Size** | 8192 | 2048-16384 | VRAM vs Speed - Lower = less memory but slower |
| **Density Threshold** | 25 | 10-50 | Surface Detection - Lower = more detail, Higher = smoother |
| **Decimation Target** | 10000 | 1000-50000 | File Size - Faces count in final mesh |
| **Smoothing Iterations** | 3 | 0-10 | Mesh Smoothness - More iterations = smoother |
| **Smoothing Lambda** | 0.5 | 0.1-1.0 | Smoothing Strength - Higher = more aggressive |
| **Remesh Size** | 0.01 | 0.005-0.05 | Triangle Size - Smaller = finer detail |

## Files Modified

### 1. `tsr/system.py`
- **Modified**: `extract_mesh()` method signature
- **Added**: New parameters: `remesh_size`, `decimation_target`, `s钉othing_iterations`, `smooth_lambda`
- **Effect**: Now accepts user-configurable parameters instead of hardcoded values

### 2. `gradio_app.py`
- **Added**: 7 new Gradio sliders for parameter control
- **Updated**: `generate()` function to accept and pass all parameters
- **Updated**: `run_example()` function with default values
- **Updated**: UI tips section with parameter explanations
- **Modified**: Model initialization to use dynamic chunk size

### 3. `run.py`
- **Added**: CLI arguments for all new parameters
- **Updated**: `extract_mesh()` call to pass all parameters
- **Effect**: Command-line interface now supports all parameters

## Parameter Relationships

### Performance Tuning
- **High VRAM**: Increase `chunk_size` (max 16384)
- **Low VRAM**: Decrease `chunk_size` (min 2048)

### Quality Tuning
- **Maximum Detail**: 
  - High `mc_resolution` (512)
  - Low `density_threshold` (10)
  - High `decimation_target` (50000)
  
- **Balanced Quality**:
  - Medium `mc_resolution` (256-320)
  - Medium `density_threshold` (20-30)
  - Medium `decimation_target` (10000-20000)

- **Fast Processing**:
  - Low `mc_resolution` (64-128)
  - High `density_threshold` (40-50)
  - Low `decimation_target` (1000-5000)

### Smoothness Tuning
- **Smoother Mesh**: Increase `smooth_iterations` (up to 10) and `smooth_lambda` (up to 1.0)
- **More Accurate Mesh**: Decrease `smooth_iterations` (down to 0) and `smooth_lambda` (down to 0.1)

## Marching Cubes Resolution Answer

**Can it go higher than 320?**

✅ **YES!** The code has no hard limit. The marching cubes resolution can theoretically go as high as memory allows.

- **Current UI Limit**: Increased from 320 to **512**
- **Practical Limit**: Depends on available VRAM
- **Memory Usage**: Resolution³ grid vertices (e.g., 512³ = 134M vertices)
- **Code Location**: `tsr/models/isosurface.py` (no hardcoded maximum)

The increase from 320 to 512 was implemented and tested with proper memory management.

## Usage Examples

### For Users with High-End GPUs (24GB+ VRAM)
```
mc_resolution: 512
chunk_size: 16384
density_threshold: 20
decimation_target: 50000
```

### For Users with Mid-Range GPUs (8-12GB VRAM)
```
mc_resolution: 256
chunk_size: 8192
density_threshold: 25
decimation_target: 10000
```

### For Users with Lower VRAM (4-6GB VRAM)
```
mc_resolution: 128
chunk_size: 4096
density_threshold: 30
decimation_target: 5000
```

## Backward Compatibility

✅ **Fully Backward Compatible**
- All defaults match original hardcoded values
- Existing workflows will produce identical results
- New parameters are optional (use defaults if not specified)
