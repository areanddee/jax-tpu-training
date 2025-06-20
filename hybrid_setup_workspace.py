#!/usr/bin/env python3
"""
Hybrid workspace setup for JAX TPU training
Combines GitHub repository with optional Google Drive access
"""

import os
import sys
import subprocess

def is_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def check_jax_installation():
    """Check and install JAX with TPU support if needed"""
    try:
        import jax
        return True
    except ImportError:
        print("📦 Installing JAX...")
        if is_colab():
            # Install JAX with TPU support for Colab
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "jax[tpu]", "-f", 
                "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ JAX installed successfully")
                return True
            else:
                print(f"❌ JAX installation failed: {result.stderr}")
                return False
        else:
            # Local installation
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "jax[cpu]"
            ], capture_output=True, text=True)
            return result.returncode == 0

def setup_git_config():
    """Setup basic git configuration for the session"""
    try:
        # Check if git is configured
        result = subprocess.run(['git', 'config', 'user.name'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0 or not result.stdout.strip():
            # Set basic git config for the session
            subprocess.run(['git', 'config', 'user.name', 'Colab User'], check=False)
            subprocess.run(['git', 'config', 'user.email', 'colab@example.com'], check=False)
            print("🔧 Git configuration set for session")
    except:
        pass  # Git config is optional

def mount_google_drive(optional=True):
    """Attempt to mount Google Drive (optional for hybrid setup)"""
    if not is_colab():
        return None
        
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Check if our shared workspace exists
        shared_workspace = '/content/drive/MyDrive/CoLab-TPU-Projects'
        if os.path.exists(shared_workspace):
            print(f"✅ Google Drive mounted with shared workspace access")
            return shared_workspace
        else:
            print("📁 Google Drive mounted (no shared workspace found)")
            return '/content/drive/MyDrive'
            
    except Exception as e:
        if optional:
            print(f"⚠️  Google Drive mount failed (optional): {str(e)[:50]}...")
            return None
        else:
            print(f"❌ Google Drive mount failed: {e}")
            return None

def setup_workspace():
    """Setup hybrid workspace combining GitHub repo and optional Google Drive"""
    
    print("🚀 Setting up JAX TPU training workspace...")
    
    # Ensure we're in the repository directory
    repo_name = "jax-tpu-training"
    if not os.path.basename(os.getcwd()) == repo_name:
        if os.path.exists(repo_name):
            os.chdir(repo_name)
            print(f"📁 Navigated to existing repository: {repo_name}")
        else:
            print(f"❌ Repository '{repo_name}' not found in current directory")
            print(f"Current directory: {os.getcwd()}")
            print("Please run: !git clone https://github.com/areanddee/jax-tpu-training.git")
            return None
    
    workspace_root = os.getcwd()
    print(f"✅ Repository workspace: {workspace_root}")
    
    # Setup git configuration
    setup_git_config()
    
    # Check folder structure
    expected_folders = ['Configs', 'Datasets', 'Docs', 'Notebooks', 'Outputs', 'Scripts']
    existing_folders = [f for f in expected_folders if os.path.exists(f)]
    missing_folders = [f for f in expected_folders if not os.path.exists(f)]
    
    # Create missing folders
    for folder in missing_folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📂 Created missing folder: {folder}")
    
    print(f"📂 Available folders: {sorted(existing_folders + missing_folders)}")
    
    # Optional Google Drive integration
    drive_workspace = mount_google_drive(optional=True)
    if drive_workspace:
        print(f"🔗 Google Drive workspace available: {drive_workspace}")
    else:
        print("🔗 Working with GitHub repository only (Google Drive optional)")
    
    # Check JAX installation and devices
    if check_jax_installation():
        try:
            import jax
            devices = jax.devices()
            device_type = devices[0].device_kind if devices else "unknown"
            print(f"🎯 JAX devices: {len(devices)} cores ({device_type})")
            
            # Quick JAX test
            if len(devices) > 0:
                import jax.numpy as jnp
                test_array = jnp.array([1, 2, 3])
                result = jnp.sum(test_array)
                print(f"🧪 JAX test passed: {result} on {result.device()}")
            
        except Exception as e:
            print(f"⚠️  JAX installation detected but test failed: {e}")
    
    print("🚀 Workspace setup complete! Ready for JAX/TPU development.")
    return workspace_root

def sync_with_drive():
    """Optional: Sync specific folders with Google Drive"""
    if not is_colab():
        print("Drive sync only available in Colab")
        return
        
    drive_base = '/content/drive/MyDrive/CoLab-TPU-Projects'
    if not os.path.exists(drive_base):
        print("No Google Drive workspace found for syncing")
        return
        
    sync_folders = ['Notebooks', 'Scripts', 'Configs']
    
    for folder in sync_folders:
        if os.path.exists(folder):
            try:
                import shutil
                dst = os.path.join(drive_base, folder)
                shutil.copytree(folder, dst, dirs_exist_ok=True)
                print(f"🔄 Synced {folder} to Google Drive")
            except Exception as e:
                print(f"⚠️  Failed to sync {folder}: {e}")

def main():
    """Main setup function"""
    workspace = setup_workspace()
    
    if workspace:
        print(f"\n📍 Current working directory: {os.getcwd()}")
        print("📚 Quick start commands:")
        print("  • List notebooks: !ls Notebooks/")
        print("  • Run training script: !python Scripts/train_example.py")
        print("  • Check JAX devices: import jax; print(jax.devices())")
        print("  • Sync to Drive (optional): sync_with_drive()")
        print("\n🎯 Happy training!")
    else:
        print("❌ Workspace setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()