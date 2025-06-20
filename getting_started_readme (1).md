# JAX TPU Training - Getting Started

Welcome to the JAX TPU training environment! This guide will help you set up Google Colab with TPU/CPU support and access to our shared workspace.

## Quick Start (2 Steps)

1. **Open Colab and clone this repo**
2. **Run the setup script**

That's it! You'll have access to JAX, TPUs (when available), and our shared workspace.

## Detailed Setup Instructions

### Step 1: Access Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Create a new notebook

### Step 2: Choose Your Runtime

**For TPU access (recommended when available):**
- Runtime → Change runtime type → Hardware accelerator → **TPU** → Save
- Note: TPUs have usage limits and may not always be available

**For CPU fallback:**
- Runtime → Change runtime type → Hardware accelerator → **None** → Save
- Always available, good for learning JAX basics

### Step 3: Clone Repository and Setup

Copy and run this code in your first cell:

```python
# Clone the training repository
!git clone https://github.com/areanddee/jax-tpu-training.git
%cd jax-tpu-training

# Run the hybrid workspace setup
%run setup_workspace.py
```

### Step 4: Verify Setup

You should see output like:
```
✅ Repository cloned successfully
✅ Connected to workspace: /content/jax-tpu-training
📂 Available folders: ['Configs', 'Datasets', 'Docs', 'Notebooks', 'Outputs', 'Scripts']
🎯 JAX devices: 8 cores (TPU v2)  # or "1 cores (cpu)"
🚀 Ready for JAX/TPU development!
```

## Understanding the Workspace

### Repository Structure

```
jax-tpu-training/
├── Configs/          # Training configurations
├── Datasets/         # Small datasets and data loading scripts
├── Docs/             # Documentation and guides
├── Notebooks/        # Jupyter notebooks with examples
├── Outputs/          # Training results and logs
├── Scripts/          # Python training scripts
└── setup_workspace.py # Workspace setup script
```

### Hybrid Workspace

We use a **hybrid approach** combining:

- **GitHub Repository**: Version-controlled code, notebooks, and configurations
- **Google Drive**: Large datasets, model outputs, and shared resources (when access is provided)

The folder structure is identical in both systems for consistency.

## Common Tasks

### Running Example Notebooks

```python
# Navigate to notebooks folder
%cd Notebooks

# List available examples
!ls *.ipynb

# Run a specific notebook
%run example_jax_basics.ipynb
```

### Training a Model

```python
# Navigate to scripts folder
%cd Scripts

# Run a training script
!python train_basic_model.py --config ../Configs/basic_training.yaml
```

### Checking Available Hardware

```python
import jax
print(f"Available devices: {jax.devices()}")
print(f"Device count: {len(jax.devices())}")
print(f"Device type: {jax.devices()[0].device_kind}")
```

## Troubleshooting

### TPU Not Available
```
Cannot connect to TPU backend
```
**Solution**: TPUs have usage limits. Try:
1. Use CPU runtime instead (Runtime → Change runtime type → None)
2. Try again later when TPU usage is lower
3. Consider upgrading to Colab Pro for more reliable TPU access

### Repository Clone Fails
```
fatal: destination path 'jax-tpu-training' already exists
```
**Solution**: Repository already cloned. Just run:
```python
%cd jax-tpu-training
%run setup_workspace.py
```

### JAX Import Error
```
ModuleNotFoundError: No module named 'jax'
```
**Solution**: Install JAX (the setup script should handle this):
```python
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### Google Drive Access Issues
**Note**: Google Drive integration is optional. The repository works completely standalone. Drive access is only needed for large shared datasets.

## Best Practices

### Saving Your Work

1. **Notebooks**: Save frequently (Ctrl+S) - they auto-save to your Google Drive
2. **Code changes**: Consider creating your own branch:
   ```python
   !git checkout -b your-name-experiments
   !git add .
   !git commit -m "My experimental changes"
   ```

### Session Management

- **Colab sessions timeout after ~12 hours of inactivity**
- **Runtime changes (CPU ↔ TPU) restart the session and lose variables**
- **Files in `/content/` are temporary** - use Git or Google Drive for persistence

### Resource Usage

- **TPU cores**: 8 cores available when TPU runtime is active
- **CPU cores**: 2 cores on free tier
- **RAM**: 12GB on free tier, 25GB on Colab Pro
- **Disk**: 100GB temporary storage per session

## Getting Help

### Learning Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Tutorials](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Colab TPU Tutorial](https://colab.research.google.com/notebooks/tpu.ipynb)

### Team Support

- Check the `Docs/` folder for team-specific guides
- Review example notebooks in `Notebooks/` folder
- Ask questions in team communication channels

### Common JAX Patterns

```python
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap

# Basic JAX operations
key = random.PRNGKey(42)
x = random.normal(key, (1000, 1000))
y = jnp.dot(x, x.T)
print(f"Computation ran on: {y.device()}")

# JIT compilation for speed
@jit
def fast_function(x):
    return jnp.dot(x, x.T)

result = fast_function(x)
```

## Next Steps

1. **Explore the notebooks**: Start with `Notebooks/01_jax_basics.ipynb`
2. **Run example training**: Try `Scripts/train_simple_model.py`
3. **Experiment**: Create your own notebooks and training scripts
4. **Share progress**: Commit useful changes back to the repository

Happy training! 🚀