# Import the Julia package manager
from juliacall import Pkg as jlPkg

# Activate the environment in the current folder
jlPkg.activate(".")

# Install Unfold (in the activated environment)
jlPkg.add("Unfold")