from eulerangles import euler2euler
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import subprocess
import argparse
import torch
import json
import os


# Rotation Matrix from ZYZ Euler Angles
def zyz_euler_to_matrix(rot, tilt, psi):
    """
    Construct rotation matrix using ZYZ Euler angles.
    Angles are in radians. Shape: (...).
    Returns: Rotation matrices of shape (..., 3, 3)
    """
    # Rz(rot)
    ca, sa = torch.cos(rot), torch.sin(rot)
    Rz1 = torch.stack([
        torch.stack([ca, -sa, torch.zeros_like(ca)], dim=-1),
        torch.stack([sa,  ca, torch.zeros_like(ca)], dim=-1),
        torch.stack([torch.zeros_like(ca), torch.zeros_like(ca), torch.ones_like(ca)], dim=-1)
    ], dim=-2)

    # Ry(tilt)
    cb, sb = torch.cos(tilt), torch.sin(tilt)
    Ry = torch.stack([
        torch.stack([cb, torch.zeros_like(cb), sb], dim=-1),
        torch.stack([torch.zeros_like(cb), torch.ones_like(cb), torch.zeros_like(cb)], dim=-1),
        torch.stack([-sb, torch.zeros_like(cb), cb], dim=-1)
    ], dim=-2)

    # Rz(psi)
    cg, sg = torch.cos(psi), torch.sin(psi)
    Rz2 = torch.stack([
        torch.stack([cg, -sg, torch.zeros_like(cg)], dim=-1),
        torch.stack([sg,  cg, torch.zeros_like(cg)], dim=-1),
        torch.stack([torch.zeros_like(cg), torch.zeros_like(cg), torch.ones_like(cg)], dim=-1)
    ], dim=-2)

    return Rz1 @ Ry @ Rz2  # Shape: (..., 3, 3)

# Conversion from zyz intrinsic to xyz extrinsic
zyz2xyz = partial(euler2euler, source_axes='zyz', source_right_handed_rotation=True, source_intrinsic=True,
    target_axes='xyz', target_right_handed_rotation=True, target_intrinsic=False, invert_matrix=False)

def append_vectors_to_pdb(pdb_path, step, energy, vectors, per_atom_min_dist=None):
    """
    Appends a new MODEL to a PDB file using the given vectors.
    File can be loaded in ChimeraX with 'open file.pdb coordset true'

    Parameters:
        pdb_path (str): File path to the PDB file.
        vectors (torch.Tensor or np.ndarray): Shape (N, 3, M)
        step (int): Optimization step. Used directly as MODEL number.
        energy (float): Repulsion energy value.
        per_atom_min_dist (torch.Tensor or np.ndarray): min distance to any other atom
    """
    import numpy as np
    import torch
    import os

    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()

    assert vectors.shape == (N, 3, M), \
        "Expected input shape (N, 3, M)"

    if per_atom_min_dist is not None:
        if isinstance(per_atom_min_dist, torch.Tensor):
            per_atom_min_dist = per_atom_min_dist.detach().cpu().numpy()
        assert per_atom_min_dist.shape == (N, 1, M), \
            "Expected input shape (N, 1, M)"

    with open(pdb_path, 'a') as f:
        f.write(f"MODEL     {step:>6d}\n")
        f.write(f"REMARK {args.lossfunc} loss: {energy:.10f}\n")

        atom_num = 1
        for unit_idx, unit in enumerate(vectors):  # unit shape: (3, M)
            unit = unit.T  # shape: (M, 3)
            if per_atom_min_dist is not None:
                unit_min_dist = per_atom_min_dist[unit_idx].reshape((M, )) # unit_min_dist shape: (M,)
            for atom_idx, (x, y, z) in enumerate(unit):
                atom_name = atom_names[atom_idx % M]  # wrap in case of overrun
                res_num = unit_idx + 1
                r = unit_min_dist[atom_idx] if per_atom_min_dist is not None else 0

                # Careful about the format of this, pdb is sensitive to number of spaces
                # https://github.com/haddocking/pdb-tools/blob/master/pdbtools/pdb_validate.py
                # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
                f.write(  
                    f"HETATM{atom_num:>5d}  {atom_name:<3} {'VEC':>3} A{res_num:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{r:6.2f}{0:6.2f}"
                    f"{'':>10}{'C':>2}  \n"
                    )
                atom_num += 1
        f.write("ENDMDL\n")

def repulsion_energy(vectors, lossfunc):
    """
    Compute total repulsion energy and per-atom min distance to any other atom.
    
    Args:
        vectors: Tensor of shape (N, 3, M)
    
    Returns:
        energy (scalar),
        per_atom_min_dist (tensor of shape (N, 1, M))
    """
    N, _, M = vectors.shape
    total_atoms = N * M

    # Prepare points
    points = vectors.transpose(1, 2).reshape(total_atoms, 3)
    points = points / points.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Pairwise distances
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # (NM, NM, 3)
    dist = diff.norm(dim=-1).clamp(min=1e-12)  # (NM, NM)

    # Use upper triangle indices to avoid duplicate/self pairs
    i, j = torch.triu_indices(dist.size(0), dist.size(1), offset=1)

    # Total energy of only unique pairs
    if lossfunc == 'log':
        # See the loss function in the 7th problem in https://en.wikipedia.org/wiki/Smale%27s_problems
        total_energy = -torch.log(dist[i, j]).sum()
    elif lossfunc == 'thomson':
        # See https://en.wikipedia.org/wiki/Thomson_problem
        total_energy = (1.0 / dist[i, j]).sum()
    else:
        raise NotImplementedError('Choose one of {"log", "thomson"}.')

    # Replace diagonal distances (distance from self) with infinity
    eye = torch.eye(total_atoms, dtype=torch.bool, device=points.device)
    dist[eye] = torch.inf

    # Smallest dist per atom to get a better undrestanding of which might be repelled the most
    per_atom_min_dist = dist.min(dim=1).values.view(N, 1, M)

    return total_energy, per_atom_min_dist

def color_atoms(N):
    """
    Generates ChimeraX color commands for N units (vector sets) using the Viridis colormap.
    Each unit corresponds to a residue group like :1, :2, ..., :N.
    """
    command = ''
    
    # Get N evenly spaced colors from the Viridis colormap
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(N - 1, 1)) for i in range(N)]  # RGBA tuples

    for unit_idx, rgba in enumerate(colors):
        r, g, b, _ = rgba
        r255, g255, b255 = int(r * 255), int(g * 255), int(b * 255)
        command += f'color :{unit_idx + 1} rgb({r255},{g255},{b255}); '

    # # ALTERNATIVE COLORING - not really nice, keeping anyways (comment out coloring above)
    # # Coloring fixed XYZ axes bright reg, green, blue and rest XYZ with less saturation
    # command = ''
    # for unit_idx in range(N):
    #     for axis, hue in zip(('X', 'Y', 'Z'), (0, 120, 240)):
    #         # 0 = red, 120 = green, 240 = red
    #         saturation = 100 if unit_idx == 0 else  35 # 100 // (unit_idx+1)
    #         command += f'color @C{axis}N@C{axis}P&:{unit_idx+1} hsl({hue}, {saturation}%, 60%); '

    return command

def render_4_scenes(pdb_path, N, n_frames, best_loss, best_step, best_loss_thomson):
    """
    Returns one long ChimeraX command composed of many small commands separated by ;
    This particular set of commands loads the pdb file as coordset, displays atoms
    as spheres on white background and over 4 scenes it shows the fixed atoms, 
    initialization of N vector sets, optimization, and final solution.
    """
    mention_M = f', M={M}' if M > 1 else ''
    additional_metric = f'/ thomson loss: {best_loss_thomson:.9f}.' if args.lossfunc != 'thomson' else ''
    initial_atom_r = 0.2 if (N * M) < 50 else 10 / (N * M)
    final_atom_r = initial_atom_r * 3 if (N * M) < 50 else initial_atom_r * 4

    return (
        # Setting up the scene
        'shape sphere radius 1.0; '  # This will be model #1
        f'open {pdb_path} coordset true; '  # This makes ChimeraX aware that each model in pdb is actually a time step
        'set bgColor white; '
        'style #2 sphere; '
        f'size #2 atomRadius {final_atom_r}; '  # First setting this to what the final value will be so orient has the info
        'lighting full; '
        'graphics silhouettes true; '
        'windowsize 800;'  # width, height in pixels. If only one int is given, only width will be adjusted
        'view orient; '  # Use orient after windowsize, otherwise the model does not have to fit into the view
        f'size #2 atomRadius {initial_atom_r}; '  # Setting this to smaller size so we see individual atoms roaming on the sphere.
        f'{color_atoms(N)}'
        'hide all; '
        'show :1; '
        'coordset #2 0; '  # Go to the first frame of the coord set
        'movie record; '  # Start recording the movie

        # First scene
        f"2dlab text 'N={N}{mention_M} Fixed atoms on XYZ poles' size 26 x 0.03 y 0.95; "  # Will be added to models with #2
        'wait 30; '  # Render 30 frames
        'close #3 ;'  # Close the previous text
        'movie crossfade frames 30; '  # Crossfade linearlly over the next 30 frames

        # Second scene
        f"2dlab text 'N={N}{mention_M} Initial position' size 26 x 0.03 y 0.95; "  # Will be added to models with #2
        'show all; '
        'wait 30; '  # Render 30 frames
        'close #4 ;'  # Close the previous text
        'movie crossfade frames 30; '  # Crossfade linearlly over the next 30 frames
        
        # Third scene
        f"2dlab text 'N={N}{mention_M} Optimizing for {best_step}/{args.steps} steps...' size 26 x 0.03 y 0.95; "  # Will be added to models with #2
        'wait 30; '  # Render 30 frames
        'coordset #2 0, pauseFrames 2; '  # Make a video from #1 using frames 0 to all, with pauseFrames render each frame N times (making the video longer)
        f'wait {int(2 * n_frames * 1.0)}; '  # In parallel to previous command, wait this many frames before starting with the next command
        'close #5 ;'  # Close the previous text
        'movie crossfade frames 30; '  # Crossfade linearlly over the next 30 frames

        # Fourth scene
        f"2dlab text 'N={N}{mention_M} Best {args.lossfunc} loss: {best_loss:.9f} {additional_metric}' size 26 x 0.03 y 0.95; "
        'wait 30; '
        'turn y 2 180; '  # turn the whole model around y axis in 2 degree increment 180 times (360 deg)
        'wait 180; '  # In parallel to previous command, Wait this many frames before starting with the next command 
        'movie crossfade frames 10; '  # Crossfade linearlly over the next N frames
        'close #6; '
        'color #2 #e5a50aff; '
        'wait 10; '
        # Gradually make the atomRadius larger from 0.2 to 0.6 with 0.01 increment per frame
        f"{''.join([f'movie crossfade frames 2; size #2 atomRadius {initial_atom_r + (0.01 * (i+1)):.2f}; wait 1; ' for i in range(int((final_atom_r - initial_atom_r) // 0.01) + 1)])}"
        'turn y 2 180; '  # turn the whole model around y axis in 2 degree increment 180 times (360 deg)
        'wait 180; '  # In parallel to previous command, Wait this many frames before starting with the next command
        'wait 30; ' # Final static scene
        f'movie encode {pdb_path.replace(".pdb", ".mp4")} framerate 30.0 quality {args.video_quality};'  # highest | higher | high | good | medium | fair | low 
        )

def save_json_report(filename, N, M, base_vectors, steps_completed, best_loss, best_step, best_eulers, best_v, best_loss_thomson):

    # Collect configuration details
    log_data = {
        "config": {
            "N": N,
            "M": M,
            "steps": args.steps,
            "loss_function": args.lossfunc,
            "learning_rate": args.lr,
            "seed": args.seed,
            "device": args.device,
            "snapshot_step": args.snapshot_step,
            "min_grad_norm": args.min_grad_norm,
            "early_stopping": args.early_stopping,
            },
        "results": {
            "steps_completed": steps_completed,
            "best_loss": float(best_loss),
            "best_loss_thomson": float(best_loss_thomson),  # Storing thomson loss for comparison with other methods
            "best_step": best_step,
            "best_euler_ZYZ_intrinsic_deg": [],
            "best_euler_XYZ_extrinsic_deg": [],
            "best_vectors": [],  # The first unit is the base_vectors
            } 
        }

    # Gather Euler angles
    for i in range(N):
        rot, tilt, psi = np.rad2deg(best_eulers[i])
        log_data["results"]["best_euler_ZYZ_intrinsic_deg"].append(
            {"rot": float(rot), "tilt": float(tilt), "psi": float(psi)})
        x, y, z = zyz2xyz(np.rad2deg(best_eulers[i]))
        log_data["results"]["best_euler_XYZ_extrinsic_deg"].append(
            {"x": float(x), "y": float(y), "z": float(z)})

    # Format the vector units so they can be stored in json
    for unit_idx, unit in enumerate(best_v):  # Shape (N, 3, M)
        for vec_idx, vec in enumerate(unit.T):  # Transpose to shape (M, 3)
            log_data["results"]["best_vectors"].append({
                "unit": f"Unit_{unit_idx + 1}",
                "vector_index": vec_idx,
                "x": float(vec[0]),
                "y": float(vec[1]),
                "z": float(vec[2])
            })

    # Write to JSON
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=4)

# Model: Trainable Euler Angles for N-1 units (first unit is fixed)
class RotationOptimizer(nn.Module):
    def __init__(self, N):
        super().__init__()
        # Initialize to zeros (not good for the optimizer)
        # self.euler_angles = nn.Parameter(torch.zeros(N - 1, 3))  # (N-1, 3)
        
        # Initialize to small random noise
        # TODO, implement initialization with previous N-1 solution
        self.euler_angles = nn.Parameter(0.05 * torch.randn(N - 1, 3))

    def forward(self):
        rot, tilt, psi = self.euler_angles[:, 0], self.euler_angles[:, 1], self.euler_angles[:, 2]
        R_trainable = zyz_euler_to_matrix(rot, tilt, psi)  # (N-1, 3, 3)

        # Prepend identity matrix for the first unit
        identity = torch.eye(3, device=R_trainable.device).unsqueeze(0)  # (1, 3, 3)
        return torch.cat([identity, R_trainable], dim=0)  # (N, 3, 3)

# Training Loop
def optimize_rotations(base_vectors, N, M, steps=1000, lr=1e-2):

    dropped_snapshots = 0
    snapshots = {}  # Here we will store intermediate results under the step key.

    model = RotationOptimizer(N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Tracking for early stopping
    best_loss, stuck_for = np.inf, 0

    for step in range(steps):
        optimizer.zero_grad()

        # Get rotation matrices (N, 3, 3)
        R = model()  # Forward pass

        # Apply rotations to base_vectors: result (N, 3, M)
        rotated = torch.einsum('nij,jm->nim', R, base_vectors)

        # Compute repulsion energy loss
        loss, per_atom_min_dist = repulsion_energy(rotated, args.lossfunc)

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Steps not important for the optimization itself.
        with torch.no_grad():

            # Track the snapshot with minimal energy found so far
            if loss.item() < best_loss:
                best_step = step
                best_loss = loss.item()
                best_eulers = torch.cat([torch.zeros(1, 3, device=device), model.euler_angles.detach()], dim=0).cpu().numpy()
                best_v = rotated.detach().numpy()
                best_d = per_atom_min_dist
                stuck_for = 0
                is_stuck_message = '( 📉 )'
            else:
                stuck_for += 1
                is_stuck_message = f'({stuck_for:>4d})'

            # Early stopping (stop if loss did not improve over the best_loss for early_stopping amount of steps)
            if args.early_stopping:
                if stuck_for >= args.early_stopping:
                    print(f'❌ Stopping early at step {step:04d} because the best loss did not improve for {stuck_for} steps.')
                    break

            # Save a snapshot if grad norm is large enough and inform the user
            if step % args.snapshot_step == 0 or step == steps - 1:

                # Drop the snapshots if the grad norm is too small (prevents videos from showing static frames)
                grad_norm = model.euler_angles.grad.norm().item()
                if grad_norm <= args.min_grad_norm:
                    is_dropped_message = '🟥 dropping'
                    dropped_snapshots += 1
                else:
                    is_dropped_message = '🟩 saving  '
                    snapshots[step] = (loss.item(), rotated, per_atom_min_dist)

                # Final message shown to the user in the terminal
                print(f"Step {step:>07d} | Loss: {loss.item():14.6f} {is_stuck_message} | Grad Norm: {grad_norm:14.6f} {is_dropped_message}" )

    # Store the best snapshot if it was not already
    snapshots[best_step] = (best_loss, best_v, best_d)  # Making sure the best snapshot is stored

    print(f"\nBest configuration found at step {best_step} optimizing the {args.lossfunc} loss. Best loss: {best_loss}.")

    # Compute the thomson loss even if it was not used for optimization (for comparison with other methods)
    if args.lossfunc != 'thomson':
        best_loss_thomson, _ = repulsion_energy(torch.from_numpy(best_v).to(args.device), lossfunc='thomson')
        best_loss_thomson = best_loss_thomson.item()
        print(f"Best configuration (thomson loss): {best_loss_thomson}.")
    else:
        best_loss_thomson = best_loss

    # Creating time-stamped filenames
    filename = os.path.join(args.out_folder, f"M={M}_N={N}_TL={best_loss_thomson:.9f}_Seed={seed}_{timestamp}.json")
    pdb_path = filename.replace('.json', '.pdb')

    # Export all kept snapshots up to the best one into a pdb file
    n_frames = 0
    for key in sorted(snapshots.keys()):
        if key > best_step:
            break
        loss, vectors, min_dist = snapshots[key]
        append_vectors_to_pdb(pdb_path, step=key, energy=loss, vectors=vectors, per_atom_min_dist=min_dist)
        n_frames += 1

    # Track input hyperparameters, some info on optimization, and final results.
    save_json_report(filename, N, M, base_vectors, step, best_loss, best_step, best_eulers, best_v, best_loss_thomson)

    # Render a ChimeraX video using the snapshots stored in the pdb
    if args.render:
        chimerax_cmd = render_4_scenes(pdb_path, N, n_frames, best_loss, best_step, best_loss_thomson)
        print(f"Dropped {dropped_snapshots} frames due to small gradient norm.")
        print('Rendering ChimeraX video, please wait...')
        subprocess.run(["chimerax", "--cmd", chimerax_cmd, '--exit'], check=True)

    print(f"\nResults saved to {filename},.pdb,{'.mp4' if args.render else ''}.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize rotations of N units of vectors for minimal repulsion energy.")
    parser.add_argument('--N', type=int, default=5, help='Number of vectors units. Specify the unit manually in the code.')
    parser.add_argument('--steps', type=int, default=20000, help='Number of optimization steps.')
    parser.add_argument('--lossfunc', type=str, default='log', help='Whether to use `log`: -log(r) or `thomson`: 1/r loss.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the Adam optimizer.')
    parser.add_argument('--snapshot_step', type=int, default=100, help='Save vectors to pdb each N optimization steps and inform user in the terminal.')
    parser.add_argument('--early_stopping', type=int, default=0, help='Stop if best loss did not improve for N steps.')
    parser.add_argument('--min_grad_norm', type=float, default=0.01, help='If gradient norm is bellow this threshold, the snapshot is considered static.')
    parser.add_argument('--drop_static', action=argparse.BooleanOptionalAction, default=False, help='Do not save static snapshots to prevent long videos.')
    parser.add_argument('--render', action=argparse.BooleanOptionalAction, default=False, help='Output a video.')
    parser.add_argument('--video_quality', type=str, default='good', help='One of {"highest", "higher", "high", "good", "medium", "fair", "low"}.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use, either cpu or cuda.')
    parser.add_argument('--out_folder', type=str, default='outputs', help='Folder to store the output files into.')

    # Capturing the start time for naming the files
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    args = parser.parse_args()
    os.makedirs(args.out_folder, exist_ok=True)


    ##################################################
    ##################################################
    # CONFIG - CHOOSE ONE USECASE OR CREATE YOUR OWN #
    ##################################################
    ##################################################

    # Define one set of M atoms that will be fixed (M=1 would be a standard Thomson problem)
    # WARNING!!! 
    # Make sure each atom name starts with a character ChimeraX recognizes as a valid atom, e.g. C, O, N.
    # Make sure each atom name is at most 3 letters long. Do not use special characters.


    # ### Usecase 1 ####################################
    # # Here we use M=1, i.e. standard Thomson problem.
    # # I use the Z unit vector so it shows in the center of the screen as a starting point.
    # atom_names = ('C', )
    # M_vectors_df = pd.DataFrame({
    #     'X': [0.0],
    #     'Y': [0.0],
    #     'Z': [1.0]
    # }, index=atom_names)
    # ##################################################

    ### Usecase 2 ###################################
    #################################################
    # Here we use M=6 atoms located where X, Y, Z axes cross the 2-sphere in both positive and negative direction.
    # Atom names are chosen as follows: C for carbon so ChimeraX can read it, X Y Z for axes names, P for positive direction, N for negative.
    atom_names = ('CXP', 'CYP', 'CZP', 'CXN', 'CYN', 'CZN')  
    M_vectors_df = pd.DataFrame({
        'X': [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
        'Y': [0.0, 1.0, 0.0, 0.0, -1.0, 0.0],
        'Z': [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
    }, index=atom_names)
    #################################################


    ##################################################
    # END OF CONFIG ##################################
    ##################################################
    

    # Fix seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Checks
    assert args.steps > 0, 'Choose valid amount of steps.'
    assert args.early_stopping >= 0, 'Choose positive int for early_stopping.'

    # Settings
    device = torch.device(args.device)
    N = args.N
    M = len(M_vectors_df)
    base_vectors = torch.tensor(M_vectors_df.values.T, dtype=torch.float32, device=device)  # Shape: (3, M)

    print(f'Running computation with N={N}, steps={args.steps}, lr={args.lr}, seed={args.seed} on device={args.device}.')
    optimize_rotations(base_vectors, N=N, M=M, steps=args.steps, lr=args.lr)

    print("\nFINISHED")
