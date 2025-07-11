# Thomson Problem Numerical Solver

![git_header](https://github.com/user-attachments/assets/1a603931-fe18-4e2a-a456-ee2538c3fb2d)

This project implements a gradient-based numerical solver for the classical [**Thomson problem**](https://en.wikipedia.org/wiki/Thomson_problem), generalized to support **arbitrary particle units** with fixed internal spatial relationships.

Instead of treating each particle as an independent point on the unit sphere, this solver allows each unit to be a set of M particles, arranged in a user-defined configuration relative to one another. These particles are rigidly linked: their internal angles and distances are preserved throughout optimization. This is achieved by rotating the entire unit (set of vectors) together as a rigid body.


## Key Features

* **Generalized Units:** Each unit is defined by a set of M vectors originating from the origin. These vectors describe the relative positions of the unit's particles.
* **Rigid Units:** During optimization, each unit rotates as a whole, keeping the internal geometry fixed.
* **Energy Minimization:** The solver minimizes a repulsion-based energy function across all particles on the sphere.
* **Support for Classical Thomson Problem:** The classical case is simply M = 1, where each unit is a single particle.
* **Generates a Trajectory File:** Outputs a `.pdb` file logging the transformed vectors and energy after each iteration.
* **Outputs Found Rotation Matrices:** The final rotation matrices for each unit are stored in both ZYZ intrinsic and XYZ extrinsic formats.
* **Renders a ChimeraX Video Visualization:** Outputs an `.mp4` video visualizing the initialization, optimization, and the end result.


## Example Use Case

For instance, a unit could represent:

* A classical single point charge (M = 1).
* A symmetric vector set of six particles aligned along the ±X, ±Y, and ±Z axes (M = 6), useful for repeatedly slicing 3D space along orthogonal planes with varying orientations, minimizing overlap between individual slices.
* A custom molecule-like configuration with an arbitrary, fixed internal spatial structure, enabling optimization of complex composite units rather than individual particles.


## Disclaimer

This solver is intended as an exploratory and educational tool for studying the Thomson problem and its generalizations. I also used it as an opportunity to get more familiar with scripting the movie renders in ChimeraX. The following limitations and caveats apply:

* **No claim of superiority:** This implementation does not claim to outperform or improve upon existing gradient-based solvers for the classical Thomson problem.
* **Novelty not guaranteed:** While this solver may be among the first to address the variant involving rigid multi-particle units, almost no literature serach has been conducted yet to confirm its novelty. Check https://arxiv.org/html/2506.08398v1 for most recent article we found on this topic.
* **No guarantee of global optimality:** The solver performs local gradient-based optimization. It may find locally optimal configurations that could be globally optimal, but this is not guaranteed. In fact, identifying global optima in this setting relates to [Smale’s 7th unsolved mathematical problem](https://en.wikipedia.org/wiki/Smale%27s_problems).
* **Sensitivity to hyperparameters:** The outcome depends on hyperparameters such as learning rate, initialization, and optimizer choice. Different settings can lead to different local optima. However, random seeds are consistently set to ensure that results are reproducible under identical configurations.

Use this tool with awareness of these limitations, especially when interpreting the quality or uniqueness of the computed solutions.


## Install

These installation commands were written and tested on Ubuntu 24.04. This program should just as well run on Windows or Mac - adjust the following if you want to use different virtual environment or package manager.

```bash
# Clone the repository (naturally assuming you have Git installed)
git clone https://gitlab.com/paloha/thomson.git
cd thomson

# Create a virtual env (using in-built Python3 virtualenv manager)
python3 -m virtualenv .venv
source .venv/bin/activate

# Install dependencies (this is a freeze of a working environment for CPU)
pip3 install -r requirements.txt

# In case you plan to use video rendering, make sure to have UCSF ChimeraX program installed and available from commandline
# Confirm by running the following command, if it does not return anything, chimerax is not installed
which chimerax
```

List of dependencies if you want to make your own environment:
```
eulerangles==1.0.2
matplotlib===3.10.3
torch==2.71
pandas==2.3.0
numpy==2.3.1
```


## Usage

Edit `thomson.py` in a text editor and provide an (M, 3) array of vectors. 
Each vector defines the position of a particle in the unit relative to the origin. 
The optimization will preserve this structure by rotating all vectors in the unit together.

``` bash
python thomson.py --N 32 --steps 20000 --lossfunc thomson --lr 5e-2 --snapshot_step 30 --early_stopping 250 --min_grad_norm 0.07 --drop_static --render --video_quality good --seed 0 --device cpu --out_folder outputs
```

The command above will output the following files:
```bash
outputs/M=1_N=32_TL=412.261260986_Seed=0_YYYY-MM-DD_HH:MM:SS.json  # Containing human readable config and results
outputs/M=1_N=32_TL=412.261260986_Seed=0_YYYY-MM-DD_HH:MM:SS.pdb  # File openable in ChimeraX - use command `open file.pdb coordsets true`
outputs/M=1_N=32_TL=412.261260986_Seed=0_YYYY-MM-DD_HH:MM:SS.mp4  # Rendered video
```

https://github.com/user-attachments/assets/08e5964e-31ae-4607-b78b-c04b0ea29bec

## Acknowledgement

I would like to thank my friend and a mathematician Dennis Elbrächter who pointed out that my original problem was related Thomson problem, which was the impulse to create this repository.


