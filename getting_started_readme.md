# JAX TPU Training

JAX and TPU training materials for team onboarding using Google Colab with hybrid GitHub + Google Drive workflow.

## Quick Demo - Try This First!

### Option 1: Run the Complete Demo
```python
# In a new Colab notebook:
!git clone https://github.com/areanddee/jax-tpu-training.git
%cd jax-tpu-training
%run setup_workspace_hybrid.py
%run Scripts/RK4v_jax_v2.py
```

### Option 2: Open Existing Demo Notebook
1. Clone this repository in Colab
2. Open `Notebooks/Hybrid-Jax-Demo.ipynb`
3. Run all cells

### What You'll See
✅ **Hybrid workspace setup** - GitHub + Google Drive integration  
✅ **JAX vectorized solver** - Runge-Kutta for dy/dt = y  
✅ **Beautiful exponential curves** - Multiple initial conditions solved simultaneously  
✅ **Hardware flexibility** - Same code on CPU or TPU

## Full Team Setup

### For Team Members

1. **Enable TPU** (when available):
   - Runtime → Change runtime type → Hardware accelerator → TPU → Save
   - *Note: TPUs may not always be available on free tier*

2. **Run setup script**:
   ```python
   !git clone https://github.com/areanddee/jax-tpu-training.git
   %cd jax-tpu-training  
   %run setup_workspace_hybrid.py
   ```

3. **Start learning JAX!**

## Environment Setup

### TPU vs CPU

**TPU (Recommended when available):**
- 8 TPU v2 cores for fast training
- Optimized for large matrix operations
- May have usage limits on free Colab

**CPU (Always available):**
- Good for learning JAX fundamentals
- Unlimited availability
- Slower for large computations

### What Gets Set Up

**Hybrid Workspace Structure:**

```
📁 GitHub (Code & Notebooks)     📁 Google Drive (Data & Outputs)
├── Configs/                     ├── Configs/
├── Datasets/                    ├── Datasets/          
├── Docs/                        ├── Docs/
├── Notebooks/                   ├── Notebooks/
├── Outputs/                     ├── Outputs/
└── Scripts/                     └── Scripts/
```

**GitHub**: Version-controlled code, notebooks, configs
**Google Drive**: Large datasets, model outputs, shared resources

## Detailed Setup Instructions

### First Time Setup

1. **Get Repository Access**
   - This is a public repo - no special access needed
   - For private company data, request access to shared Google Drive folder

2. **Open Google Colab**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Sign in with your Google account

3. **Create New Notebook**
   - File → New Notebook
   - Or open our starter notebook (link above)

4. **Choose Runtime**
   ```
   Runtime → Change runtime type
   
   For TPU training:
   - Hardware accelerator: TPU
   - Runtime shape: Standard
   
   For CPU learning:
   - Hardware accelerator: None
   - Runtime shape: Standard
   ```

5. **Run Setup Script**
   ```python
   # Clone repository and setup environment
   !git clone https://github.com/areanddee/jax-tpu-training.git
   %cd jax-tpu-training
   %run setup_workspace_hybrid.py
   ```

### What the Setup Script Does

1. **Clones GitHub repository** with all training code
2. **Mounts Google Drive** for shared data access (if available)
3. **Installs JAX** with TPU support
4. **Configures workspace** with proper folder structure
5. **Verifies hardware** (TPU/CPU detection)

### Working with the Hybrid System

**For Code Development:**
- Edit notebooks and scripts in the GitHub folders
- Commit changes: `!git add . && git commit -m "your message" && git push`

**For Data and Results:**
- Large datasets go in Google Drive `/Datasets/`
- Model outputs save to Google Drive `/Outputs/`
- Share results via Google Drive folders

**Best Practices:**
- Keep code in GitHub (< 100MB files)
- Keep data in Google Drive (large files)
- Use same folder names for easy navigation

## Common Issues & Solutions

### "Cannot connect to TPU backend"
- **Solution**: Click "Connect without TPU" and use CPU for learning
- **Alternative**: Try again during off-peak hours
- **Upgrade**: Consider Colab Pro for reliable TPU access

### "Google Drive mount failed"
- **Solution**: Restart runtime and try again
- **Alternative**: Work without Google Drive (uses local storage)
- **Workaround**: Manual file upload/download as needed

### "Repository not found"
- **Solution**: Make sure you have internet connection
- **Check**: Repository is public and accessible
- **Alternative**: Download ZIP and upload to Colab manually

### Lost work after session timeout
- **Prevention**: Save notebooks to Google Drive frequently (Ctrl+S)
- **Recovery**: Re-run setup script to restore environment
- **Best practice**: Commit code changes to GitHub regularly

## Learning Path

### Beginner: Start with the Demo
1. **Run**: `Notebooks/Hybrid-Jax-Demo.ipynb` - See the system working
2. **Explore**: `Scripts/RK4v_jax_v2.py` - JAX vectorized numerical methods
3. **Learn**: JAX basics with working examples

### Intermediate: JAX Fundamentals  
1. Start with `Notebooks/01_jax_basics.ipynb` (when available)
2. Learn array operations and transformations
3. Practice with CPU (TPU not required)

### Advanced: TPU Optimization
1. Run `Notebooks/02_tpu_introduction.ipynb` (when available)
2. Learn TPU-specific patterns
3. Experiment with batch sizes and precision

## Folder Guide

### Configs/
- Training configurations and experiment parameters
- TPU-specific settings and hardware configs

### Notebooks/
- `Hybrid-Jax-Demo.ipynb` - **Start here!** Complete working demo
- Learning tutorials and examples
- Save your experimental notebooks here

### Scripts/
- `RK4v_jax_v2.py` - JAX vectorized Runge-Kutta solver demo
- `setup_workspace_hybrid.py` - Main setup script
- Production training scripts and utilities

### Datasets/
- Sample datasets for tutorials
- Links to larger datasets in Google Drive
- Data preprocessing scripts

### Outputs/
- Model checkpoints and weights
- Training logs and metrics
- Generated visualizations

### Docs/
- Additional documentation
- Team resources and guides
- Reference materials

## Support

- **Technical Issues**: Create an issue in this repository
- **Access Problems**: Contact project maintainer
- **JAX Questions**: Check [JAX documentation](https://jax.readthedocs.io/)
- **Colab Help**: See [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## Contributing

1. Work on your assigned notebooks/scripts
2. Test on both CPU and TPU when possible
3. Commit changes with descriptive messages
4. Share results in Google Drive `/Outputs/` folder
5. Document new techniques in `/Docs/`

---

**Happy JAX Training! 🚀**