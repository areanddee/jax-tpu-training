#!/usr/bin/env python3
"""
Hybrid JAX TPU Training Workspace Setup
Combines GitHub repository with Google Drive data storage
"""

import os
import sys
import subprocess
import shutil

def is_colab():
    """Detect if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def run_command(cmd, description=""):
    """Run shell command with error handling"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Warning: {description} failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def setup_github_repo():
    """Clone or update GitHub repository"""
    repo_url = "https://github.com/areanddee/jax-tpu-training.git"
    repo_name = "jax-tpu-training"
    
    print("📡 Setting up GitHub repository...")
    
    # Check if we're already inside the repo
    current_dir = os.path.basename(os.getcwd())
    if current_dir == repo_name and os.path.exists('.git'):
        print(f"✅ Already in repository {repo_name}")
        print("📝 Skipping git operations (already in repo)")
        return True
    
    # Check if repo exists as subdirectory
    if os.path.exists(repo_name):
        print(f"📁 Repository {repo_name} already exists")
        os.chdir(repo_name)
        print("📝 Skipping git operations (existing repo)")
        return True
    
    # Repo doesn't exist, need to clone
    print(f"📥 Cloning repository from {repo_url}")
    if run_command(f"git clone {repo_url}", "git clone"):
        print("✅ Repository cloned successfully")
        os.chdir(repo_name)
        return True
    else:
        print("❌ Failed to clone repository")
        return False
    
    # If not in repo, proceed with original logic
    if os.path.exists(repo_name):
        print(f"📁 Repository {repo_name} already exists")
        os.chdir(repo_name)
        
        if run_command("git pull origin main", "git pull"):
            print("✅ Repository updated to latest version")
        else:
            print("⚠️  Could not update repository (continuing with existing version)")
    else:
        print(f"📥 Cloning repository from {repo_url}")
        if run_command(f"git clone {repo_url}", "git clone"):
            print("✅ Repository cloned successfully")
            os.chdir(repo_name)
        else:
            print("❌ Failed to clone repository")
            return False
    
    # Configure git for this session
    run_command('git config user.name "Colab User"', "git config name")
    run_command('git config user.email "user@colab.com"', "git config email")
    
    return True

def setup_google_drive():
    """Mount Google Drive for data access"""
    if not is_colab():
        print("💻 Local environment - Google Drive not needed")
        return True
    
    print("🔗 Setting up Google Drive access...")
    
    try:
        from google.colab import drive
        
        # Check if already mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive already mounted")
            return True
        
        # Mount Google Drive
        drive.mount('/content/drive')
        
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive mounted successfully")
            return True
        else:
            print("❌ Google Drive mount failed")
            return False
            
    except Exception as e:
        print(f"⚠️  Google Drive setup failed: {e}")
        print("📝 Continuing without Google Drive (local storage only)")
        return False

def verify_folder_structure():
    """Verify workspace folder structure"""
    expected_folders = ['Configs', 'Datasets', 'Docs', 'Notebooks', 'Outputs', 'Scripts']
    current_folders = [f for f in os.listdir('.') if os.path.isdir(f)]
    
    missing_folders = []
    for folder in expected_folders:
        if folder not in current_folders:
            missing_folders.append(folder)
    
    if missing_folders:
        print(f"📂 Creating missing folders: {', '.join(missing_folders)}")
        for folder in missing_folders:
            os.makedirs(folder, exist_ok=True)
    
    print(f"📋 Workspace folders: {sorted([f for f in os.listdir('.') if os.path.isdir(f)])}")
    return True

def setup_jax():
    """Install and verify JAX with TPU support"""
    print("🔧 Setting up JAX...")
    
    try:
        import jax
        print(f"✅ JAX already installed (version {jax.__version__})")
    except ImportError:
        print("📦 Installing JAX with TPU support...")
        if is_colab():
            # Colab-specific JAX installation
            install_cmd = "pip install -q jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        else:
            # Local installation
            install_cmd = "pip install -q jax[cpu]"
        
        if run_command(install_cmd, "JAX installation"):
            print("✅ JAX installed successfully")
        else:
            print("❌ JAX installation failed")
            return False
    
    # Verify JAX and detect hardware
    try:
        import jax
        devices = jax.devices()
        device_type = devices[0].device_kind
        device_count = len(devices)
        
        if device_type == 'TPU':
            print(f"🎯 JAX detected: {device_count} TPU cores (v2)")
        elif device_type == 'GPU':
            print(f"🎯 JAX detected: {device_count} GPU(s)")
        else:
            print(f"🎯 JAX detected: {device_count} CPU core(s)")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX verification failed: {e}")
        return False

def show_workspace_info():
    """Display workspace information"""
    print("\n" + "="*50)
    print("🚀 WORKSPACE READY!")
    print("="*50)
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"📂 Available folders: {sorted([f for f in os.listdir('.') if os.path.isdir(f)])}")
    
    # Show Google Drive status
    if is_colab() and os.path.exists('/content/drive/MyDrive/CoLab-TPU-Projects'):
        print(f"☁️  Google Drive: Connected to shared workspace")
    elif is_colab():
        print(f"💾 Google Drive: Not connected (using local storage)")
    else:
        print(f"💻 Local environment: Using filesystem storage")
    
    # Show next steps
    print("\n📚 Next Steps:")
    print("   1. Start with: Notebooks/01_jax_basics.ipynb")
    print("   2. Check hardware: import jax; print(jax.devices())")
    print("   3. Save work: Ctrl+S to save notebook")
    print("   4. Commit code: !git add . && git commit -m 'your message'")
    print("="*50)

def main():
    """Main setup function"""
    print("🛠️  JAX TPU Training Workspace Setup")
    print("🔄 Hybrid: GitHub + Google Drive")
    print("-" * 40)
    
    # Step 1: Setup GitHub repository
    if not setup_github_repo():
        print("❌ Setup failed: Could not access GitHub repository")
        return False
    
    # Step 2: Setup Google Drive (optional)
    setup_google_drive()
    
    # Step 3: Verify folder structure
    verify_folder_structure()
    
    # Step 4: Setup JAX
    if not setup_jax():
        print("❌ Setup failed: JAX installation/verification failed")
        return False
    
    # Step 5: Show workspace info
    show_workspace_info()
    
    return True

def check_google_drive_sync():
    """Check if Google Drive sync is available and show sync options"""
    if not is_colab():
        return
    
    drive_project_path = '/content/drive/MyDrive/CoLab-TPU-Projects'
    if os.path.exists(drive_project_path):
        print(f"\n🔄 Google Drive sync available:")
        print(f"   GitHub workspace: {os.getcwd()}")
        print(f"   Google Drive workspace: {drive_project_path}")
        print(f"   Use: sync_to_drive() function for manual sync")

def sync_to_drive():
    """Helper function to sync GitHub folders to Google Drive"""
    if not is_colab():
        print("💻 Sync only available in Colab environment")
        return
    
    github_base = os.getcwd()
    drive_base = '/content/drive/MyDrive/CoLab-TPU-Projects'
    
    if not os.path.exists(drive_base):
        print("❌ Google Drive workspace not found")
        return
    
    sync_folders = ['Notebooks', 'Scripts', 'Configs']
    
    for folder in sync_folders:
        src = os.path.join(github_base, folder)
        dst = os.path.join(drive_base, folder)
        
        if os.path.exists(src):
            try:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"✅ Synced {folder} to Google Drive")
            except Exception as e:
                print(f"⚠️  Failed to sync {folder}: {e}")
    
    print("🔄 Sync to Google Drive complete!")

if __name__ == "__main__":
    success = main()
    if success:
        check_google_drive_sync()
        print("\n💡 Tip: Run sync_to_drive() to copy notebooks to Google Drive")
    else:
        print("\n❌ Setup incomplete - please check errors above")
        sys.exit(1)
