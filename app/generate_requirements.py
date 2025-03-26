import subprocess
import sys
import os

def generate_requirements_file(output_file="requirements.txt"):
    """
    Generate a requirements.txt file containing all installed packages
    in the current virtual environment with their versions.
    
    Args:
        output_file (str): The name of the output requirements file.
    """
    # Get list of installed packages using pip freeze
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        packages = result.stdout.splitlines()
        
        # Sort packages alphabetically (case-insensitive)
        packages.sort(key=lambda x: x.lower())
        
        # Write packages to requirements file
        with open(output_file, 'w') as f:
            f.write('\n'.join(packages) + '\n')
        
        print(f"Requirements file generated at: {os.path.abspath(output_file)}")
        print(f"Total packages: {len(packages)}")
    
    except subprocess.CalledProcessError as e:
        print(f"Error running pip freeze: {e}")
        sys.exit(1)

if __name__ == "__main__":
    generate_requirements_file()