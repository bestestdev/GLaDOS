import argparse
import os
from pathlib import Path
import platform
from shutil import which
import subprocess
import shutil


def is_uv_installed() -> bool:
    """
    Check if the UV tool is installed on the system.

    Returns:
        bool: True if the 'uv' command is available in the system path, False otherwise.
    """
    return which("uv") is not None


def install_uv() -> None:
    """
    Install the UV package management tool across different platforms.

    This function checks if UV is already installed. If not, it performs the installation:
    - On Windows, it uses pip to install UV and then updates it
    - On other platforms, it uses a curl-based installation script from Astral.sh

    Raises:
        subprocess.CalledProcessError: If the installation or update commands fail
    """
    if is_uv_installed():
        print("UV is already installed")
        return

    print("Installing UV...")
    if platform.system() == "Windows":
        subprocess.run(["pip", "install", "uv"])
        subprocess.run(["uv", "self", "update"])
    else:
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)
        subprocess.run(["uv", "self", "update"])


def setup_config_files() -> None:
    """
    Set up configuration files by copying examples if the target files don't exist.
    
    This ensures users have working configuration files without needing to manually copy them.
    """
    config_dir = Path("configs")
    
    # List of example files to copy
    example_files = list(config_dir.glob("*.yaml.example"))
    
    for example_file in example_files:
        target_file = config_dir / example_file.name.replace(".example", "")
        
        if not target_file.exists():
            print(f"Creating config file: {target_file}")
            shutil.copy(example_file, target_file)
        else:
            print(f"Config file already exists: {target_file}")


def download_vision_models() -> None:
    """
    Download models needed for the Vision module.
    
    This runs the Vision model downloader script as a module import rather than a direct script
    execution to avoid relative import errors.
    """
    try:
        print("Downloading Vision models...")
        
        # Use a module-based approach to run the download script
        # This avoids the relative import error when running the script directly
        venv_python = ".venv/bin/python" if not platform.system() == "Windows" else ".venv\\Scripts\\python.exe"
        result = subprocess.run(
            [venv_python, "-m", "glados.Vision.download_models"], 
            check=False
        )
        
        if result.returncode == 0:
            print("Vision models downloaded successfully")
        else:
            print(f"Failed to download Vision models: process exited with code {result.returncode}")
    except Exception as e:
        print(f"Error downloading Vision models: {e}")


def main() -> None:
    """
    Set up the project development environment by installing UV, creating a virtual environment,
    and preparing the project for development.

    This function performs the following steps:
    1. Changes the current working directory to the project root
    2. Installs the UV package management tool
    3. Creates a Python 3.12.8 virtual environment
    4. Detects CUDA availability
    5. Installs the project in editable mode with appropriate dependencies
    6. Downloads and verifies project model files
    7. Set up configuration files
    8. Downloads Vision module models if vision dependencies are installed

    The function handles different platform-specific configurations and supports both CUDA and CPU-only installations.

    Notes:
        - Requires UV package manager to be available
        - Assumes project is structured with a standard Python project layout
        - Modifies system environment variables during execution
    """
    parser = argparse.ArgumentParser(description="Set up the project development environment.")
    parser.add_argument("--api", action="store_true", help="Install API dependencies.")
    parser.add_argument("--vision", action="store_true", help="Install Vision dependencies.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Install UV
    install_uv()

    # Create virtual environment
    subprocess.run(["uv", "venv", "--python", "3.12.8"])

    # Determine if CUDA is available
    if platform.system() == "Windows":
        venv_python = ".venv/bin/python"
    else:
        venv_python = ".venv/bin/python"
        os.environ["PATH"] = f"{os.path.dirname(venv_python)}:{os.environ['PATH']}"

    try:
        has_cuda = subprocess.run(["nvcc", "--version"], capture_output=True, check=False).returncode == 0
    except FileNotFoundError:
        has_cuda = False

    extras = ["cuda"] if has_cuda else ["cpu"]
    if args.api:
        extras.append("api")
    if args.vision:
        extras.append("vision")

    # Install project in editable mode
    env = os.environ.copy()
    env["PATH"] = f"{os.path.abspath('.venv/bin')}:{env['PATH']}"
    os.environ["VIRTUAL_ENV"] = os.path.abspath(".venv")
    os.system(f"uv pip install -e .[{','.join(extras)}]")

    # Download and verify model files
    os.system("uv run glados download")
    
    # Set up config files
    setup_config_files()
    
    # Download Vision models if vision extra was installed
    if args.vision:
        download_vision_models()


if __name__ == "__main__":
    main()
