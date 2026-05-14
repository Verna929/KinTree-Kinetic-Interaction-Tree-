import MDAnalysis as mda
import numpy as np
import torch
from MDAnalysis.lib.distances import distance_array
import mdtraj as md
from collections import defaultdict
import pickle
import os
import json

atom_masks = {
    "ARG": {
        "donors": ["NE", "NH1", "NH2", "N"],
        "acceptors": ["O"],
        "cation": ["CZ", "NH1", "NH2"]
    },
    "LYS": {
        "donors": ["NZ", "N"],
        "acceptors": ["O"],
        "cation": ["NZ"]
    },
    "HIS": {
        "donors": ["ND1", "NE2", "N"],
        "acceptors": ["ND1", "NE2", "O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"]
    },
    "HID": {  # δ-protonated
        "donors": ["ND1", "N"],
        "acceptors": ["NE2", "O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"]
    },
    "HSD": {  # δ-protonated
        "donors": ["ND1", "N"],
        "acceptors": ["NE2", "O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"]
    },
    "HIE": {  # ε-protonated
        "donors": ["NE2", "N"],
        "acceptors": ["ND1", "O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"]
    },
    "HSE": {  # ε-protonated
        "donors": ["NE2", "N"],
        "acceptors": ["ND1", "O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"]
    },
    "HIP": {  # both protonated (cationic)
        "donors": ["ND1", "NE2", "N"],
        "acceptors": ["O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "cation": ["ND1", "NE2"]
    },
    "HSP": {  # both protonated (cationic)
        "donors": ["ND1", "NE2", "N"],
        "acceptors": ["O"],
        "aromatic": ["CG", "ND1", "CD2", "CE1", "NE2"],
        "cation": ["ND1", "NE2"]
    },
    "ASP": {
        "donors": ["N"],
        "acceptors": ["OD1", "OD2", "O"],
        "anion": ["OD1", "OD2"]
    },
    "ASH": {  # protonated Asp (neutral)
        "donors": ["OD1", "N"],
        "acceptors": ["OD2", "O"]
    },
    "GLU": {
        "donors": ["N"],
        "acceptors": ["OE1", "OE2", "O"],
        "anion": ["OE1", "OE2"]
    },
    "GLH": {  # protonated Glu (neutral)
        "donors": ["OE1", "N"],
        "acceptors": ["OE2", "O"]
    },
    "ASN": {
        "donors": ["ND2", "N"],
        "acceptors": ["OD1", "O"]
    },
    "GLN": {
        "donors": ["NE2", "N"],
        "acceptors": ["OE1", "O"]
    },
    "SER": {
        "donors": ["OG", "N"],
        "acceptors": ["OG", "O"]
    },
    "THR": {
        "donors": ["OG1", "N"],
        "acceptors": ["OG1", "O"]
    },
    "TYR": {
        "donors": ["OH", "N"],
        "acceptors": ["OH", "O"],
        "aromatic": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    },
    "TRP": {
        "donors": ["NE1", "N"],
        "acceptors": ["O"],
        "aromatic": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"]
    },
    "PHE": {
        "donors": ["N"],
        "acceptors": ["O"],
        "aromatic": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]
    },
    "CYS": {
        "donors": ["SG", "N"],
        "acceptors": ["SG", "O"],
        "chalcogen": ["SG"],
    },
    "CYM": {
        "donors": ["N"],
        "acceptors": ["SG","O"],
        "anion": ["SG"]
    },
    "CYX": {
        "donors": ["N"],
        "acceptors": ["O"],
        "chalcogen":["SG"]
    },
    "MET": {
        "donors": ["N"],
        "acceptors": ["O"], # SD is a weak acceptor, usually O is sufficient for backbone
        "chalcogen":["SD"]
    },
    "ALA": {
        "donors": ["N"],
        "acceptors": ["O"]
    },
    "VAL": {
        "donors": ["N"],
        "acceptors": ["O"]
    },
    "ILE": {
        "donors": ["N"],
        "acceptors": ["O"]
    },
    "LEU": {
        "donors": ["N"],
        "acceptors": ["O"]
    },
    "PRO": {
        # PRO lacks backbone NH donor
        "donors": [],
        "acceptors": ["O"]
    },
    "GLY": {
        "donors": ["N"],
        "acceptors": ["O"]
    },
}

STANDARD_AMINO_ACIDS = set(atom_masks.keys())

def user_define_ligand_masks():
    """
    Interactively asks the user to define atom masks for ligands.
    Returns a dictionary compatible with the 'atom_masks' structure.
    """
    print("=" * 60)
    print("      INTERACTIVE LIGAND MASK GENERATOR")
    print("=" * 60)
    print("Instructions: Enter atom names separated by commas (e.g., N1, N3).")
    print("If a category is empty, just press ENTER.")
    print("-" * 60)

    try:
        num_ligands = int(input("How many different ligands do you want to define? (e.g., 1): ").strip())
    except ValueError:
        print("Invalid number. Please restart and enter an integer.")
        return {}

    new_masks = {}

    for i in range(num_ligands):
        print(f"\n--- Defining Ligand {i + 1} ---")

        res_name = input("Enter 3-letter Residue Name (e.g., URC): ").strip().upper()
        if not res_name:
            print("Residue name cannot be empty!")
            continue

        def get_atom_list(prompt_text):
            raw_input = input(f"{prompt_text}: ")
            if not raw_input.strip():
                return []
            return [x.strip() for x in raw_input.split(',') if x.strip()]

        donors = get_atom_list(f"[{res_name}] Donors (e.g., N1, N7)")
        acceptors = get_atom_list(f"[{res_name}] Acceptors (e.g., O11, O13)")
        aromatic = get_atom_list(f"[{res_name}] Aromatic Ring Atoms (e.g., C4, C5, N7)")
        cation = get_atom_list(f"[{res_name}] Cationic Atoms (+)")
        anion = get_atom_list(f"[{res_name}] Anionic Atoms (-)")

        new_masks[res_name] = {
            "donors": donors,
            "acceptors": acceptors,
            "aromatic": aromatic,
            "cation": cation,
            "anion": anion
        }

        print(f"✔ Successfully added {res_name}")

    print("=" * 60)
    print("Generation Complete.")
    return new_masks

def validate_topology(topology_file, atom_masks, ignore_residues=None):
    """
    Checks if EVERY residue in the topology file has a corresponding entry
    in the atom_masks dictionary.

    Raises a ValueError if undefined residues are found.
    """
    if ignore_residues is None:
        ignore_residues = {"SOL", "HOH", "TIP3", "WAT", "NA", "CL", "K", "MG"}

    print(f"Validating topology: {topology_file}...")

    try:
        u = mda.Universe(topology_file)
    except Exception as e:
        raise ValueError(f"Could not load topology file: {e}")

    pdb_resnames = set(u.residues.resnames)

    defined_masks = set(atom_masks.keys())
    missing_residues = []

    for res in pdb_resnames:
        clean_res = res.strip().upper()

        if clean_res in ignore_residues:
            continue

        if clean_res not in defined_masks:
            missing_residues.append(clean_res)

    if missing_residues:
        print("\n" + "!" * 60)
        print("CRITICAL ERROR: MISSING ATOM MASKS")
        print("!" * 60)
        print(f"The following residues were found in '{topology_file}' but are NOT defined in your atom_masks:")
        print(f"\n   MISSING: {', '.join(missing_residues)}\n")
        print("Your model will ignore these residues completely if you proceed.")
        print("Please run 'user_define_ligand_masks()' or update the dictionary manually.")
        print("!" * 60 + "\n")

        raise ValueError(f"Undefined residues found: {missing_residues}")

    print("✔ Topology Validation Passed: All residues are defined.")
    return True

def save_masks(masks, filepath="atom_masks.json"):
    """Saves the atom_masks dictionary to a human-readable JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(masks, f, indent=4)
        print(f"Successfully saved masks to {filepath}")
    except Exception as e:
        print(f"Error saving masks: {e}")

def load_masks(filepath="atom_masks.json"):
    """Loads masks from a JSON file."""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return {}

    try:
        with open(filepath, "r") as f:
            loaded_masks = json.load(f)
        print(f"Successfully loaded {len(loaded_masks)} masks from {filepath}")
        return loaded_masks
    except Exception as e:
        print(f"Error loading masks: {e}")
        return {}

def find_max_batch_size(topology, trajectory, start=50, max_frames=None, verbose=True):
    print("Finding maximum batch size...")
    batch_size = start
    last_successful = start

    while True:
        try:
            if verbose:
                print(f"Testing batch size: {batch_size}")
            traj_iter = md.iterload(trajectory, top=topology, chunk=batch_size)
            for chunk in traj_iter:
                md.compute_contacts(chunk, scheme='closest-heavy')
                break  # Only test first chunk
            last_successful = batch_size
            batch_size *= 2  # Try next power of 2
        except Exception as e:
            if verbose:
                print(f"Batch size {batch_size} failed: {e}")
            break

    if verbose:
        print(f"✔ Maximum safe batch size: {last_successful}")

    return last_successful

def _soft_score(d, d0, n=4):
    """
    Generalized switching function, d in nm, d0 in nm.
    Vectorized for NumPy arrays.
    """
    return 1.0 / (1.0 + (d / d0) ** n)

def contact_map(topology, trajectory, ligand_names=None, threshold_nm=0.55, batch_size=100,
                outputpath='./contacts_chunked.pkl'):
    """
    Chunked contact computation.

    Output file is a pickle stream of per-chunk records:
        record = {
            "chunk_idx": int,
            "frame_start": int,  # 0-based frame index
            "n_frames": int,
            "pairs_bin": List[np.ndarray],   # len = n_frames; each (n_pairs, 2) int32
            "scores":   List[np.ndarray],    # len = n_frames; each (n_pairs, 3) float32: [i, j, score]
        }

    Returns:
        outputpath (string)
    """
    print("Start calculating contacts (Hybrid + Soft Scores)...")

    explicit_ligand_pairs = None
    if ligand_names:
        if isinstance(ligand_names, str):
            ligand_names = [ligand_names]
        print(f"Pre-calculating explicit pairs for ligands: {ligand_names}")
        temp_top = md.load_topology(topology)

        target_names = {ln.strip().upper() for ln in ligand_names}
        solvent_names = {"HOH", "SOL", "WAT", "TIP3", "TIP4P", "NA", "CL", "K", "MG"}

        ligand_indices = []
        non_solvent_indices = []
        for r in temp_top.residues:
            rname = r.name.strip().upper()
            if rname in target_names:
                ligand_indices.append(r.index)
            if rname not in solvent_names:
                non_solvent_indices.append(r.index)

        if ligand_indices:
            pairs_list = [[l, r] for l in ligand_indices for r in non_solvent_indices if l != r]
            explicit_ligand_pairs = np.asarray(pairs_list, dtype=np.int32)
            print(f"✔ Generated {len(explicit_ligand_pairs)} explicit ligand pairs (ignoring solvent).")
        else:
            print(f"⚠️ Warning: Ligands {ligand_names} not found.")

    with open(outputpath, "wb") as f_out:
        pass  # truncate/overwrite

    traj_iter = md.iterload(trajectory, top=topology, chunk=batch_size)

    frame_cursor = 0  # 0-based global frame index
    for chunk_idx, chunk in enumerate(traj_iter):
        contacts, pairs = md.compute_contacts(chunk, scheme='closest-heavy', ignore_nonprotein=True)
        if explicit_ligand_pairs is not None:
            l_contacts, l_pairs = md.compute_contacts(chunk, contacts=explicit_ligand_pairs, scheme='closest-heavy')
            if l_contacts.size:
                contacts = np.hstack([contacts, l_contacts])
                pairs = np.vstack([pairs, l_pairs])

        pairs_bin_list = []
        scores_list = []

        for fi in range(chunk.n_frames):
            dists = contacts[fi]
            mask = dists < threshold_nm
            fpairs = pairs[mask].astype(np.int32)
            pairs_bin_list.append(fpairs)

            if fpairs.size:
                fscores = _soft_score(dists[mask], d0=threshold_nm, n=4).astype(np.float32)
                arr = np.column_stack([fpairs.astype(np.float32), fscores])
                scores_list.append(arr.astype(np.float32))
            else:
                scores_list.append(np.empty((0, 3), dtype=np.float32))

        record = dict(
            chunk_idx=chunk_idx,
            frame_start=frame_cursor,
            n_frames=chunk.n_frames,
            pairs_bin=pairs_bin_list,
            scores=scores_list
        )

        with open(outputpath, "ab") as f_out:
            pickle.dump(record, f_out, protocol=pickle.HIGHEST_PROTOCOL)

        start_1based = frame_cursor + 1
        end_1based = frame_cursor + chunk.n_frames
        print(f"Chunk {chunk_idx+1}: Frames {start_1based} to {end_1based} done")

        frame_cursor += chunk.n_frames

    print("Contacts calculation finished.")
    print(f"Chunked contacts saved at {outputpath}.")
    return outputpath

def find_residue_index(topology_file, res_name):
    print(f"🔍 Loading topology: {topology_file}...")
    try:
        u = mda.Universe(topology_file)
    except Exception as e:
        print(f"❌ Error loading topology: {e}")
        return
    found_residues = [r for r in u.residues if r.resname.strip().upper() == res_name.upper()]

    print("\n" + "=" * 50)
    print(f"SEARCH RESULTS FOR '{res_name}'")
    print("=" * 50)

    if not found_residues:
        print(f"❌ Residue '{res_name}' not found in the topology.")
        print(f"Available residues: {set(r.resname for r in u.residues)}")
    else:
        for res in found_residues:
            print(f"✅ Found {res.resname} (PDB ID: {res.resid})")
            print(f"   👉 PYTHON INDEX: {res.ix}  <-- USE THIS NUMBER")
            print(f"   (Atom count: {len(res.atoms)})")
            print("-" * 30)

    return res.ix

def check_ligand_contacts(pkl_path, ligand_index, use_scores=False, max_examples=5):
    """
    Scan a chunked pickle-stream contact file (written by contact_map(..., outputpath=...))
    and report how often ligand_index appears in contact pairs.

    Parameters
    ----------
    pkl_path : str
        Path to the pickle stream file produced by contact_map().
    ligand_index : int
        MDTraj residue index of the ligand (same indexing used in explicit_ligand_pairs).
    use_scores : bool
        If True, also looks at record["scores"] and prints score examples (if present).
    max_examples : int
        Number of example partner residues to print.

    Returns
    -------
    dict with summary stats.
    """
    print(f"📂 Loading chunked contact stream: {pkl_path}")
    frames_with_ligand = 0
    total_ligand_contacts = 0
    examples = []

    total_frames = 0
    n_records = 0

    try:
        with open(pkl_path, "rb") as f:
            while True:
                try:
                    record = pickle.load(f)   # one chunk record
                except EOFError:
                    break

                n_records += 1
                if not isinstance(record, dict) or "pairs_bin" not in record:
                    raise ValueError(
                        f"Unrecognized record format in stream at record #{n_records}. "
                        f"Expected dict with key 'pairs_bin'. Got: {type(record)} keys={getattr(record, 'keys', lambda: [])()}"
                    )

                frame_start = int(record.get("frame_start", total_frames))
                n_frames = int(record.get("n_frames", len(record["pairs_bin"])))
                pairs_bin_list = record["pairs_bin"]

                # Optional
                scores_list = record.get("scores", None)

                if len(pairs_bin_list) != n_frames:
                    # be tolerant but consistent
                    n_frames = len(pairs_bin_list)

                for fi in range(n_frames):
                    global_frame = frame_start + fi
                    pairs = pairs_bin_list[fi]  # expected shape (N,2) int32

                    if pairs is None or len(pairs) == 0:
                        continue
                    pairs = np.asarray(pairs)
                    if pairs.ndim != 2 or pairs.shape[1] != 2:
                        raise ValueError(
                            f"Bad pairs_bin shape at record #{n_records}, local frame {fi}: {pairs.shape}"
                        )

                    mask = (pairs[:, 0] == ligand_index) | (pairs[:, 1] == ligand_index)
                    n_lig = int(np.sum(mask))
                    if n_lig > 0:
                        frames_with_ligand += 1
                        total_ligand_contacts += n_lig

                        # Collect examples
                        if len(examples) < max_examples:
                            contact_rows = pairs[mask]
                            for r in contact_rows:
                                partner = int(r[0] if r[1] == ligand_index else r[1])

                                if use_scores and scores_list is not None:
                                    # scores[fi] has shape (N,3): [i, j, score] as float32
                                    sc = np.asarray(scores_list[fi])
                                    if sc.size and sc.shape[1] == 3:
                                        # find matching row (order-insensitive)
                                        # NOTE: sc i,j are float32, cast to int for compare
                                        ij = sc[:, :2].astype(np.int32)
                                        m2 = ((ij[:, 0] == r[0]) & (ij[:, 1] == r[1])) | ((ij[:, 0] == r[1]) & (ij[:, 1] == r[0]))
                                        score_val = float(sc[m2, 2][0]) if np.any(m2) else None
                                    else:
                                        score_val = None
                                    examples.append((global_frame, partner, score_val))
                                else:
                                    examples.append((global_frame, partner))

                                if len(examples) >= max_examples:
                                    break

                    total_frames = max(total_frames, global_frame + 1)

    except FileNotFoundError:
        print("❌ File not found.")
        return None
    except Exception as e:
        print(f"❌ Error while scanning pickle stream: {e}")
        return None

    # --- REPORT ---
    print("\n" + "=" * 50)
    print("RESULTS (chunked contact stream)")
    print("=" * 50)
    print(f"Records (chunks) read: {n_records}")
    print(f"Total frames scanned:  {total_frames}")
    print(f"Ligand residue index:  {ligand_index}")

    if total_frames == 0:
        print("⚠️ No frames found in file (empty stream?).")
        return {
            "total_frames": 0,
            "frames_with_ligand": 0,
            "avg_contacts_per_active_frame": 0.0,
            "examples": [],
        }

    if frames_with_ligand == 0:
        print(f"❌ NO contacts found for ligand index {ligand_index}.")
        print("Potential causes:")
        print("  1) ligand_index does not match MDTraj residue indexing for this topology.")
        print("  2) ligand_names not found -> explicit_ligand_pairs was None, so no ligand-protein contacts were computed.")
        print("  3) threshold_nm too strict (try 0.6–0.8 nm as a test).")
        print("  4) You are using atom indices instead of residue indices (these pairs are RESIDUE indices).")
    else:
        frac = frames_with_ligand / total_frames * 100.0
        avg = total_ligand_contacts / frames_with_ligand
        print("✅ Ligand contacts found.")
        print(f"Frames with ligand contacts: {frames_with_ligand} / {total_frames} ({frac:.2f}%)")
        print(f"Avg contacts per active frame: {avg:.2f}")

        print("\nExample interactions:")
        if use_scores:
            for item in examples:
                fr, partner, sc = item
                print(f"  Frame {fr}: ligand {ligand_index} <-> residue {partner} | score={sc}")
        else:
            for fr, partner in examples:
                print(f"  Frame {fr}: ligand {ligand_index} <-> residue {partner}")

    return {
        "total_frames": total_frames,
        "frames_with_ligand": frames_with_ligand,
        "avg_contacts_per_active_frame": (total_ligand_contacts / frames_with_ligand) if frames_with_ligand else 0.0,
        "examples": examples,
    }

def _is_protein_backbone(res_name, atom_name):
    """
    Returns True ONLY if the atom is a backbone atom of a STANDARD amino acid.
    Ligand atoms named 'N' or 'O' will return False (kept safe).
    """
    if res_name not in STANDARD_AMINO_ACIDS:
        return False
    return atom_name in ["N", "O", "C", "CA"]


def _precompute_atom_indices(u, atom_masks):
    """
    OPTIMIZATION: Pre-calculate all atom indices once.
    This replaces slow string parsing inside the main loop.
    """
    print("Pre-computing atom indices for faster lookup...")
    cache = [None] * len(u.residues)

    for res in u.residues:
        resname = res.resname.strip().upper()
        if resname not in atom_masks:
            continue

        mask_def = atom_masks[resname]
        entry = {}

        atom_map = defaultdict(list)
        for atom in res.atoms:
            atom_map[atom.name.strip().upper()].append(atom.index)

        for category in ['donors', 'acceptors', 'anion', 'cation', 'aromatic', 'chalcogen']:
            if category in mask_def and mask_def[category]:
                indices = []
                for name in mask_def[category]:
                    found_indices = atom_map.get(name.strip().upper())
                    if found_indices:
                        indices.extend(found_indices)
                if indices:
                    entry[category] = np.array(indices, dtype=np.int64)

        cache[res.resindex] = entry
    print("✔ Atom indices cached.")
    return cache

def _ring_plane_normal(atomgroup):
    """
    Fit a plane to all atom positions in the ring using PCA.
    Returns the normal vector.
    """
    if len(atomgroup) < 3:
        raise ValueError("Need at least 3 atoms to define a plane")

    coords = atomgroup.positions
    center = coords.mean(axis=0)
    coords_centered = coords - center
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]

    return normal / np.linalg.norm(normal)

def _angle_between(v1, v2):
    """Compute angle between two vectors in degrees"""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def _can_form_any_interaction(r1, r2, has_donor, has_acceptor, is_cation, is_anion, is_aromatic, has_chalcogen):
    return (
        (r1 in has_donor and r2 in has_acceptor) or
        (r2 in has_donor and r1 in has_acceptor) or
        (r1 in is_cation and r2 in is_anion) or
        (r2 in is_cation and r1 in is_anion) or
        (r1 in is_aromatic and r2 in is_aromatic) or
        (r1 in is_cation and r2 in is_aromatic) or
        (r2 in is_cation and r1 in is_aromatic) or
        (r1 == "CYS" and r2 == "CYS") or
        (r1 == "CYX" and r2 == "CYX") or
        (r1 in has_chalcogen and r2 in has_acceptor) or
        (r2 in has_chalcogen and r1 in has_acceptor) or
        (r1 in has_chalcogen and r2 in is_aromatic) or
        (r2 in has_chalcogen  and r1 in is_aromatic)
    )

def _customized_threshold(filepath="interaction_thresholds.txt"):
    print('Start reading user customized interaction thresholds...')
    thresholds = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip comments or empty lines
            if ":" not in line:
                raise ValueError(f"Invalid line format: '{line}'")
            key, value = line.split(":", 1)
            key = key.strip()
            values = [float(v.strip()) for v in value.split(",") if v.strip()]
            if not values:
                raise ValueError(f"No threshold values provided for '{key}'")
            thresholds[key] = values if len(values) > 1 else values[0]  # single value = float

    expected_lengths = {
        "hbond": 1,
        "salt_bridge": 1,
        "pi_pi": 3,
        "T_shape": 1,
        "cation_pi": 2,
        "disulfide": 1,
        "chalcogen": 3
    }
    for key, expected in expected_lengths.items():
        if key in thresholds:
            val = thresholds[key]
            if isinstance(val, list) and len(val) != expected:
                raise ValueError(f"{key} expects {expected} values, got {len(val)}")

    return thresholds

def _is_probable_covalent_disulfide(atom1, atom2, dist, cutoff=2.3):
    """
    Avoid counting covalent CYS-SG--SG-CYS disulfide bonds as chalcogen interactions.
    Typical covalent S-S distance is around 2.0-2.1 Å.
    """
    if atom1.name.strip().upper() == "SG" and atom2.name.strip().upper() == "SG":
        if atom1.resname.strip().upper() in {"CYS", "CYX"} and atom2.resname.strip().upper() in {"CYS", "CYX"}:
            if dist < cutoff:
                return True
    return False

def _get_chalcogen_reference_atoms(residue, ch_atom):
    """
    Return atoms covalently bonded to the chalcogen atom.

    These atoms define the X-S...A angle, where:
        X = atom covalently bonded to sulfur/chalcogen
        S = chalcogen atom
        A = acceptor atom or aromatic centroid

    For rough protein interaction detection, residue-name-based references
    are sufficient.
    """
    resname = residue.resname.strip().upper()
    atom_name = ch_atom.name.strip().upper()

    ref_names = []

    if resname in {"MET", "MSE"} and atom_name in {"SD", "SE"}:
        # MET: CG-SD-CE
        ref_names = ["CG", "CE"]

    elif resname in {"CYS", "CYM", "CYX"} and atom_name == "SG":
        # CYS: CB-SG
        # For CYX, SG is also bonded to another SG, but CB is enough for rough detection.
        ref_names = ["CB"]

    refs = []
    for ref_name in ref_names:
        selected = residue.atoms.select_atoms(f"name {ref_name}")
        if len(selected) > 0:
            refs.append(selected[0])

    return refs


def _chalcogen_angle_ok(residue, ch_atom, target_pos, min_angle=150.0):
    """
    Check whether angle X-S...target is compatible with a chalcogen bond.

    target_pos can be:
        - acceptor atom position
        - aromatic ring centroid position

    Returns True if any valid X-S...target angle is >= min_angle.
    """
    refs = _get_chalcogen_reference_atoms(residue, ch_atom)

    # If no reference atom is found, fail the angle check.
    # For distance-only rough detection, do not call this function.
    if len(refs) == 0:
        return False

    s_pos = ch_atom.position

    for ref_atom in refs:
        v1 = ref_atom.position - s_pos
        v2 = target_pos - s_pos

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue

        angle = _angle_between(v1, v2)

        if angle >= min_angle:
            return True

    return False

def interaction_detect(
    topology,
    trajectory,
    filtered_pairs_all,
    customized_thresholds,
    atom_masks,
    outputpath='./interaction_detect.pkl',
    chain_mode="both",                  # "both" | "intra" | "inter"
    chain_filter_protein_only=True      # only filter protein-protein pairs by chain
):
    if chain_mode not in {"both", "intra", "inter"}:
        raise ValueError("chain_mode must be one of {'both','intra','inter'}")

    print("Start interaction detection chalcogen ver (Optimized Pre-Calculation)...")

    u = mda.Universe(topology, trajectory)
    validate_topology(topology, atom_masks)
    topology_cache = _precompute_atom_indices(u, atom_masks)

    n_frames = len(u.trajectory)
    interactions_all = [defaultdict(list) for _ in range(n_frames)]

    has_donor = {res for res, d in atom_masks.items() if "donors" in d}
    has_acceptor = {res for res, d in atom_masks.items() if "acceptors" in d}
    is_anion = {res for res, d in atom_masks.items() if "anion" in d}
    is_cation = {res for res, d in atom_masks.items() if "cation" in d}
    is_aromatic = {res for res, d in atom_masks.items() if "aromatic" in d}
    has_chalcogen = {res for res, d in atom_masks.items() if "chalcogen" in d}

    res_segid = []
    for r in u.residues:
        try:
            sid = str(r.segid).strip()
        except Exception:
            try:
                sid = str(r.segment.segid).strip()
            except Exception:
                sid = ""
        res_segid.append(sid if sid else "ALL")

    seg_to_int = {}
    res_chain_ids = np.zeros(len(u.residues), dtype=np.int32)
    for idx, sid in enumerate(res_segid):
        if sid not in seg_to_int:
            seg_to_int[sid] = len(seg_to_int)
        res_chain_ids[idx] = seg_to_int[sid]

    res_is_protein = np.array(
        [(r.resname.strip().upper() in STANDARD_AMINO_ACIDS) for r in u.residues],
        dtype=bool
    )

    all_pairs = set()
    for frame_pairs in filtered_pairs_all:
        for i, j in frame_pairs:
            all_pairs.add(tuple(sorted((int(i), int(j)))))

    valid_pair_mask = {}
    for i, j in all_pairs:
        ri = u.residues[i].resname.strip().upper()
        rj = u.residues[j].resname.strip().upper()

        if not _can_form_any_interaction(ri, rj, has_donor, has_acceptor, is_cation, is_anion, is_aromatic, has_chalcogen):
            valid_pair_mask[(i, j)] = False
        else:
            valid_pair_mask[(i, j)] = True

    thresholds = _customized_threshold(customized_thresholds)
    dist_hbond = thresholds["hbond"]
    dist_saltbridge = thresholds["salt_bridge"]
    dist_PP, angle_PP, slippage_PP = thresholds["pi_pi"]
    angle_T = thresholds["T_shape"]
    dist_cp, angle_cp = thresholds["cation_pi"]
    dist_chalcogen, dist_chalcogen_pi, angle_chalcogen = thresholds["chalcogen"]
    angle_PP_max = 180 - angle_PP
    angle_T_min = 90 - angle_T / 2
    angle_T_max = 90 + angle_T / 2
    angle_cp_max = 180 - angle_cp

    for frame_idx in range(n_frames):
        u.trajectory[frame_idx]
        frame_pairs = filtered_pairs_all[frame_idx]

        if chain_mode != "both" and len(frame_pairs) > 0:
            fp = np.asarray(frame_pairs, dtype=np.int32)
            ci = res_chain_ids[fp[:, 0]]
            cj = res_chain_ids[fp[:, 1]]
            same_chain = (ci == cj)
            keep_chain = same_chain if chain_mode == "intra" else ~same_chain

            if chain_filter_protein_only:
                prot_pair = res_is_protein[fp[:, 0]] & res_is_protein[fp[:, 1]]
                keep = (~prot_pair) | (prot_pair & keep_chain)
            else:
                keep = keep_chain

            frame_pairs = fp[keep]

        for i, j in frame_pairs:
            i = int(i); j = int(j)
            pair = tuple(sorted((i, j)))
            if not valid_pair_mask.get(pair, False):
                continue

            res_i = u.residues[i]
            res_j = u.residues[j]
            polar = False
            allow_aromatic = True
            allow_chalcogen = False

            cache_i = topology_cache[i]
            cache_j = topology_cache[j]

            if cache_i is not None and "chalcogen" in cache_i:
                allow_chalcogen = True
                #print('Potential chalcogen at residue ' + cache_i)

            if cache_j is not None and "chalcogen" in cache_j:
                allow_chalcogen = True
                #print('Potential chalcogen at residue ' + cache_j)

            # --- HYDROGEN BOND ---
            if cache_i is not None and cache_j is not None and 'donors' in cache_i and 'acceptors' in cache_j:
                donors = u.atoms[cache_i['donors']]
                acceptors = u.atoms[cache_j['acceptors']]
                if len(donors) > 0 and len(acceptors) > 0:
                    dists = distance_array(donors.positions, acceptors.positions)
                    rows, cols = np.where(dists < dist_hbond)
                    if len(rows) > 0:
                        for r, c in zip(rows, cols):
                            d_atom = donors[r]
                            a_atom = acceptors[c]
                            bb_d = _is_protein_backbone(res_i.resname, d_atom.name)
                            bb_a = _is_protein_backbone(res_j.resname, a_atom.name)
                            if bb_d and bb_a:
                                continue
                            interactions_all[frame_idx]["hbond"].append(pair)
                            polar = True
                            break

            if not polar and cache_i is not None and cache_j is not None and 'donors' in cache_j and 'acceptors' in cache_i:
                donors = u.atoms[cache_j['donors']]
                acceptors = u.atoms[cache_i['acceptors']]
                if len(donors) > 0 and len(acceptors) > 0:
                    dists = distance_array(donors.positions, acceptors.positions)
                    rows, cols = np.where(dists < dist_hbond)
                    if len(rows) > 0:
                        for r, c in zip(rows, cols):
                            d_atom = donors[r]
                            a_atom = acceptors[c]
                            bb_d = _is_protein_backbone(res_j.resname, d_atom.name)
                            bb_a = _is_protein_backbone(res_i.resname, a_atom.name)
                            if bb_d and bb_a:
                                continue
                            interactions_all[frame_idx]["hbond"].append(pair)
                            polar = True
                            break

            # --- SALT BRIDGE ---
            if polar:
                if cache_i is not None and cache_j is not None and 'anion' in cache_i and 'cation' in cache_j:
                    anion_atoms = u.atoms[cache_i['anion']]
                    cation_atoms = u.atoms[cache_j['cation']]
                    allow_aromatic = False
                    if len(anion_atoms) > 0 and len(cation_atoms) > 0:
                        if np.any(distance_array(anion_atoms.positions, cation_atoms.positions) < dist_saltbridge):
                            interactions_all[frame_idx]["salt_bridge"].append(pair)

                if allow_aromatic and cache_i is not None and cache_j is not None and 'anion' in cache_j and 'cation' in cache_i:
                    anion_atoms = u.atoms[cache_j['anion']]
                    cation_atoms = u.atoms[cache_i['cation']]
                    allow_aromatic = False
                    if len(anion_atoms) > 0 and len(cation_atoms) > 0:
                        if np.any(distance_array(anion_atoms.positions, cation_atoms.positions) < dist_saltbridge):
                            interactions_all[frame_idx]["salt_bridge"].append(pair)

            # --- AROMATIC / CATION-PI / etc. ---
            #--- AROMATIC INTERACTIONS (FIXED SYMMETRY) ---
            if allow_aromatic:
                if 'aromatic' in cache_i and 'aromatic' in cache_j:
                    ring_a = u.atoms[cache_i['aromatic']]
                    ring_b = u.atoms[cache_j['aromatic']]

                    if len(ring_a) >= 3 and len(ring_b) >= 3:
                        norm_a = _ring_plane_normal(ring_a)
                        norm_b = _ring_plane_normal(ring_b)
                        if norm_a is not None and norm_b is not None:
                            center_a = ring_a.positions.mean(axis=0)
                            center_b = ring_b.positions.mean(axis=0)
                            vec_ab = center_b - center_a
                            dist = np.linalg.norm(vec_ab)

                            # CRITICAL: Distance check happens first
                            if dist < dist_PP:
                                angle = _angle_between(norm_a, norm_b)

                                # --- SYMMETRIC GEOMETRY CHECK ---
                                # Check 1: Ring A is the reference plane (Equivalent to Loop Iteration 1)
                                proj_a = vec_ab - np.dot(vec_ab, norm_a) * norm_a
                                slip_a = np.linalg.norm(proj_a)

                                # Check 2: Ring B is the reference plane (Equivalent to Loop Iteration 2)
                                vec_ba = -vec_ab
                                proj_b = vec_ba - np.dot(vec_ba, norm_b) * norm_b
                                slip_b = np.linalg.norm(proj_b)

                                # Pi-Pi (Parallel)
                                if angle < angle_PP or angle > angle_PP_max:
                                    # If EITHER geometry fits, it's a hit (matches old code behavior)
                                    if slip_a < slippage_PP or slip_b < slippage_PP:
                                        interactions_all[frame_idx]["pi_pi"].append(pair)

                                # T-Shape (Perpendicular)
                                if angle_T_min < angle < angle_T_max:
                                    # If EITHER geometry fits, it's a hit (matches old code behavior)
                                    if slip_a < slippage_PP or slip_b < slippage_PP:
                                        interactions_all[frame_idx]["T-shape"].append(pair)

                # Cation-Pi (i=cation, j=aromatic)
                if 'cation' in cache_i and 'aromatic' in cache_j:
                    cats = u.atoms[cache_i['cation']]
                    ring = u.atoms[cache_j['aromatic']]
                    if len(cats) > 0 and len(ring) >= 3:
                        norm = _ring_plane_normal(ring)
                        if norm is not None:
                            center = ring.positions.mean(axis=0)
                            for c_pos in cats.positions:
                                vec = c_pos - center
                                if np.linalg.norm(vec) < dist_cp:
                                    angle = _angle_between(vec, norm)
                                    if angle < angle_cp or angle > angle_cp_max:
                                        interactions_all[frame_idx]["cation_pi"].append(pair)
                                        break

                # Cation-Pi (j=cation, i=aromatic)
                if 'cation' in cache_j and 'aromatic' in cache_i:
                    cats = u.atoms[cache_j['cation']]
                    ring = u.atoms[cache_i['aromatic']]
                    if len(cats) > 0 and len(ring) >= 3:
                        norm = _ring_plane_normal(ring)
                        if norm is not None:
                            center = ring.positions.mean(axis=0)
                            for c_pos in cats.positions:
                                vec = c_pos - center
                                if np.linalg.norm(vec) < dist_cp:
                                    angle = _angle_between(vec, norm)
                                    if angle < angle_cp or angle > angle_cp_max:
                                        interactions_all[frame_idx]["cation_pi"].append(pair)
                                        break
            # --- CHALCOGEN BONDS ---
            if allow_chalcogen:

                if cache_i is not None and cache_j is not None and 'chalcogen' in cache_i and 'acceptors' in cache_j:
                    ch_atoms = u.atoms[cache_i['chalcogen']]
                    acc_atoms = u.atoms[cache_j['acceptors']]

                    if len(ch_atoms) > 0 and len(acc_atoms) > 0:
                        dmat = distance_array(ch_atoms.positions, acc_atoms.positions)
                        rows, cols = np.where(dmat < dist_chalcogen)

                        for r, c in zip(rows, cols):
                            ch_atom = ch_atoms[r]
                            acc_atom = acc_atoms[c]
                            dist = dmat[r, c]

                            # Avoid counting covalent disulfide SG-SG as chalcogen
                            if _is_probable_covalent_disulfide(ch_atom, acc_atom, dist):
                                continue

                            if not _chalcogen_angle_ok(
                                    res_i,
                                    ch_atom,
                                    acc_atom.position,
                                    min_angle=angle_chalcogen
                            ):
                                continue

                            interactions_all[frame_idx]["chalcogen"].append(pair)
                            break

                if cache_i is not None and cache_j is not None and 'chalcogen' in cache_j and 'acceptors' in cache_i:
                    ch_atoms = u.atoms[cache_j["chalcogen"]]
                    acc_atoms = u.atoms[cache_i["acceptors"]]

                    if len(ch_atoms) > 0 and len(acc_atoms) > 0:
                        dmat = distance_array(ch_atoms.positions, acc_atoms.positions)
                        rows, cols = np.where(dmat < dist_chalcogen)

                        for r, c in zip(rows, cols):
                            ch_atom = ch_atoms[r]
                            acc_atom = acc_atoms[c]
                            dist = dmat[r, c]

                            # Avoid counting covalent disulfide SG-SG as chalcogen
                            if _is_probable_covalent_disulfide(ch_atom, acc_atom, dist):
                                continue

                            if not _chalcogen_angle_ok(
                                    res_j,
                                    ch_atom,
                                    acc_atom.position,
                                    min_angle=angle_chalcogen
                            ):
                                continue

                            interactions_all[frame_idx]["chalcogen"].append(pair)
                            break

                if cache_i is not None and cache_j is not None and 'chalcogen' in cache_i and 'aromatic' in cache_j:
                    ch_atoms = u.atoms[cache_i['chalcogen']]
                    ring = u.atoms[cache_j['aromatic']]

                    if len(ch_atoms) > 0 and len(ring) >= 3:
                        center = ring.positions.mean(axis=0)

                        for ch_atom in ch_atoms:
                            dist = np.linalg.norm(ch_atom.position - center)

                            if dist < dist_chalcogen_pi:
                                if not _chalcogen_angle_ok(
                                        res_i,
                                        ch_atom,
                                        center,
                                        min_angle=angle_chalcogen
                                ):
                                    continue

                                interactions_all[frame_idx]["chalcogen"].append(pair)
                                break

                if cache_i is not None and cache_j is not None and 'chalcogen' in cache_j and 'aromatic' in cache_i:
                    ch_atoms = u.atoms[cache_j['chalcogen']]
                    ring = u.atoms[cache_i['aromatic']]

                    if len(ch_atoms) > 0 and len(ring) >= 3:
                        center = ring.positions.mean(axis=0)

                        for ch_atom in ch_atoms:
                            dist = np.linalg.norm(ch_atom.position - center)

                            if dist < dist_chalcogen_pi:
                                if not _chalcogen_angle_ok(
                                        res_j,
                                        ch_atom,
                                        center,
                                        min_angle=angle_chalcogen
                                ):
                                    continue

                                interactions_all[frame_idx]["chalcogen"].append(pair)
                                break

        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx + 1}/{n_frames} interaction detection done.")

    with open(outputpath, "wb") as f:
        pickle.dump(interactions_all, f)
    print(f"Interaction information saved at {outputpath}.")
    return interactions_all


def _generate_input(contact_map, interaction_detect):
    with open(contact_map, "rb") as f1:
        contact = pickle.load(f1)
    with open(interaction_detect, "rb") as f2:
        interactions = pickle.load(f2)

    interaction_types = ['contact', 'hbond', 'salt_bridge', 'pi_pi', 'T-shape', 'cation_pi', 'chalcogen']
    interaction_to_index = {k: i for i, k in enumerate(interaction_types)}

    frames_edge_attr = []

    for frame_idx in range(len(contact)):
        filtered_pairs = contact[frame_idx]
        interaction = interactions[frame_idx]

        edge_to_attr = defaultdict(lambda: [0] * len(interaction_types))
        for i, j in filtered_pairs:
            key = tuple(sorted((i, j)))
            edge_to_attr[key][interaction_to_index['contact']] = 1
        for kind, pairs in interaction.items():
            if kind not in interaction_to_index:
                continue
            for i, j in pairs:
                key = tuple(sorted((i, j)))
                if key not in edge_to_attr:
                    print(f"[Frame {frame_idx}] Warning: interaction found without contact — {key} for {kind}")
                    edge_to_attr[key] = [0] * len(interaction_types)
                edge_to_attr[key][interaction_to_index[kind]] = 1

        edge_index = []
        edge_attr = []
        for (i, j), attr in edge_to_attr.items():
            edge_index.append([i, j])
            edge_attr.append(attr)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.uint8)
        frames_edge_attr.append((edge_index, edge_attr))

    return frames_edge_attr

def generate_input(contact_map_path, interaction_detect_path,outputpath):
    print('Start generating torch input file...')
    print('Reading previously saved information...')
    with open(contact_map_path, "rb") as f1:
        contact = pickle.load(f1)
    with open(interaction_detect_path, "rb") as f2:
        interactions = pickle.load(f2)

    interaction_types = ['contact', 'hbond', 'salt_bridge', 'pi_pi', 'T-shape', 'cation_pi', 'chalcogen']
    interaction_to_index = {k: i for i, k in enumerate(interaction_types)}

    print('Collecting unique residue pairs...')
    global_edge_set = set()
    for frame_pairs in contact:
        for i, j in frame_pairs:
            global_edge_set.add(tuple(sorted((i, j))))
    global_edges = sorted(global_edge_set)
    edge_to_index = {pair: idx for idx, pair in enumerate(global_edges)}
    edge_index = torch.tensor(global_edges, dtype=torch.long).t().contiguous()  # shape (2, N_edges)

    print('Building per frame tensor...')
    edge_attr_all = []
    for frame_idx in range(len(contact)):
        edge_attr = torch.zeros(len(global_edges), len(interaction_types), dtype=torch.uint8)

        # Mark contact
        for i, j in contact[frame_idx]:
            idx = edge_to_index[tuple(sorted((i, j)))]
            edge_attr[idx][interaction_to_index['contact']] = 1

        # Mark interactions
        for kind, pairs in interactions[frame_idx].items():
            if kind not in interaction_to_index:
                continue
            for i, j in pairs:
                idx = edge_to_index[tuple(sorted((i, j)))]
                edge_attr[idx][interaction_to_index[kind]] = 1

        edge_attr_all.append(edge_attr)

    print('Saving input file...')
    torch.save({
        "edge_index": edge_index,
        "edge_attr_all": edge_attr_all,
        "interaction_types": ['contact', 'hbond', 'salt_bridge', 'pi_pi', 'T-shape', 'cation_pi', 'chalcogen']
    }, outputpath)

    return edge_index, edge_attr_all  # edge_index is global; edge_attr_all is per frame

def _build_global_edge_set_from_interactions(interactions, interaction_types):
    """
    Collect all unique residue pairs that form any (non-contact) interaction.
    Works with both (i,j) and ((i,j), score) formats.
    """
    global_edge_set = set()
    for frame_interaction in interactions:
        for kind in interaction_types:
            for item in frame_interaction.get(kind, []):
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                    (i, j), _ = item
                else:
                    i, j = item
                global_edge_set.add(tuple(sorted((i, j))))

    global_edges = sorted(global_edge_set)
    edge_to_index = {pair: idx for idx, pair in enumerate(global_edges)}
    edge_index = torch.tensor(global_edges, dtype=torch.long).t().contiguous()
    return edge_index, edge_to_index


def generate_interaction_input(interaction_detect_path, outputpath,
                              interaction_types=['hbond', 'salt_bridge', 'pi_pi', 'T-shape', 'cation_pi', 'chalcogen'],
                              sample_for_edge_set=None):
    """
    Build PyG input from interaction_detect.pkl, storing soft scores if available.
    """
    print('Start generating torch input file (NO contacts)...')
    print('Reading interaction information...')
    with open(interaction_detect_path, "rb") as f2:
        interactions = pickle.load(f2)  # List of interaction dicts per frame

    if isinstance(interactions, dict) and "interactions_all" in interactions:
        interactions = interactions["interactions_all"]
    elif isinstance(interactions, list):
        pass
    else:
        raise ValueError("Unrecognized format in interaction_detect_path")

    interaction_to_index = {k: i for i, k in enumerate(interaction_types)}

    n_frames = len(interactions)
    if sample_for_edge_set is not None:
        print('Collecting unique residue pairs from random subset of frames...')
        if sample_for_edge_set > n_frames:
            sample_for_edge_set = n_frames
        sample_indices = sorted(torch.randperm(n_frames)[:sample_for_edge_set].tolist())
        sampled_interactions = [interactions[i] for i in sample_indices]

        edge_index, edge_to_index = _build_global_edge_set_from_interactions(sampled_interactions, interaction_types)
        print(f'Found {edge_index.size(1)} unique interaction-forming residue pairs across {sample_for_edge_set} sampled frames.')

    else:
        edge_index, edge_to_index = _build_global_edge_set_from_interactions(interactions, interaction_types)
        print(f'Found {edge_index.size(1)} unique interaction-forming residue pairs across all frames.')

    print('Building per-frame tensor...')
    edge_attr_all = []
    for frame_idx, frame_interaction in enumerate(interactions):
        edge_attr = torch.zeros(len(edge_to_index), len(interaction_types), dtype=torch.float32)
        for kind, pairs in frame_interaction.items():
            if kind not in interaction_to_index:
                continue
            for item in pairs:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], tuple):
                    (i, j), score = item
                else:
                    i, j = item
                    score = 1.0
                idx = edge_to_index.get(tuple(sorted((i, j))))
                if idx is not None:
                    edge_attr[idx][interaction_to_index[kind]] = float(score)
        edge_attr_all.append(edge_attr)

    print('Saving input file...')
    torch.save({
        "edge_index": edge_index,
        "edge_attr_all": edge_attr_all,
        "interaction_types": interaction_types
    }, outputpath)

    print(f"Saved {outputpath}: {len(edge_attr_all)} frames, {edge_index.size(1)} unique edges, {len(interaction_types)} features.")
    return edge_index, edge_attr_all

def check_ligand_status(pkl_path, ligand_index):
    print(f"🔍 Reading {pkl_path}...")

    try:
        with open(pkl_path, "rb") as f:
            interactions_all = pickle.load(f)
    except FileNotFoundError:
        print("❌ File not found. Did the calculation finish successfully?")
        return

    total_frames = len(interactions_all)
    frames_with_ligand = 0
    type_counts = defaultdict(int)

    example_interaction = None

    print(f"Scanning {total_frames} frames for Ligand Index {ligand_index}...")

    for frame_i, frame_data in enumerate(interactions_all):
        found_in_this_frame = False

        for int_type, pairs in frame_data.items():
            for (i, j) in pairs:
                # Check if 467 is either part of the pair
                if i == ligand_index or j == ligand_index:
                    type_counts[int_type] += 1
                    found_in_this_frame = True

                    if example_interaction is None:
                        partner = j if i == ligand_index else i
                        example_interaction = (frame_i, int_type, partner)

        if found_in_this_frame:
            frames_with_ligand += 1

    # --- REPORT ---
    print("\n" + "="*50)
    print(f"RESULTS FOR LIGAND {ligand_index}")
    print("="*50)

    if frames_with_ligand == 0:
        print("❌ NO INTERACTIONS DETECTED.")
        print("Troubleshooting:")
        print("1. Is '467' the 0-based Python index or the PDB Residue Number?")
        print("   (If 467 is the PDB number, the index is likely 466 or lower).")
        print("2. Did the 'atom_masks' dictionary contain the ligand entry?")
    else:
        print(f"✅ Success! Ligand interacts in {frames_with_ligand} frames ({frames_with_ligand/total_frames*100:.1f}%).")
        print("\nInteraction Counts (Total events across all frames):")
        for k, v in type_counts.items():
            print(f"  • {k:<12}: {v}")

        if example_interaction:
            frame, kind, partner = example_interaction
            print(f"\nExample: Frame {frame}, {kind} with Residue {partner}")

def load_all_filtered_pairs(pkl_path):
    filtered_pairs_all = []

    with open(pkl_path, "rb") as f:
        while True:
            try:
                record = pickle.load(f)
            except EOFError:
                break

            filtered_pairs_all.extend(record["pairs_bin"])

    return filtered_pairs_all

def load_and_filter_contacts(pkl_path, important_residues):
    """
    Loads chunked contact data, filters for important residues,
    and returns a list of lists of tuples [(i, j, score), ...].
    """
    important_arr = np.array(list(set(important_residues)))

    filtered_contacts_all = []
    total_frames = 0

    print(f"Reading and filtering from: {pkl_path}")

    with open(pkl_path, "rb") as f:
        while True:
            try:
                record = pickle.load(f)
            except EOFError:
                # End of file reached
                break

            for frame_arr in record['scores']:
                if frame_arr.size == 0:
                    filtered_contacts_all.append([])
                    continue

                i_idx = frame_arr[:, 0].astype(int)
                j_idx = frame_arr[:, 1].astype(int)
                mask = np.isin(i_idx, important_arr) | np.isin(j_idx, important_arr)
                filtered_data = frame_arr[mask]
                frame_contacts = [
                    (int(row[0]), int(row[1]), float(row[2]))
                    for row in filtered_data
                ]

                filtered_contacts_all.append(frame_contacts)

            total_frames += record["n_frames"]
            print(f"Processed chunk {record['chunk_idx']+1} (Total frames: {total_frames})", end='\r')

    print(f"\nDone. Loaded {len(filtered_contacts_all)} frames.")
    return filtered_contacts_all
