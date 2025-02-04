import sys
import os
import numpy as np
import h5py
from glob import glob
from collections import defaultdict
import yaml
from pathlib import Path
from wsi_core.WholeSlideImage import WholeSlideImage


def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, "r") as stream:
            cfg = yaml.safe_load(stream)
        return cfg
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)


def collect_paths(cfg):
    """
    Collect paths for scores, slides, and patches based on the configuration.
    Returns a dictionary where keys are slide IDs and values are lists of paths.
    """
    dic_paths = defaultdict(list)

    # Collect score paths assuming the format: <n_fold>_<slide_id>.npy
    for path in glob(os.path.join(cfg['scores_path'], f"{cfg['n_fold']}_*")):
        slide = Path(path).stem.split('.')[0][2:]
        dic_paths[slide].append(path)

    # Collect slide paths
    for path in glob(cfg['slides_path']):
        slide = Path(path).stem
        if slide in dic_paths:
            dic_paths[slide].append(path)

    # Collect patch paths
    for path in glob(os.path.join(cfg['patches_path'], '*')):
        slide = Path(path).stem
        if slide in dic_paths:
            dic_paths[slide].append(path)

    return dic_paths


def generate_heatmaps(dic_paths, cfg):
    """
    Generate heatmaps for each slide using scores and patch coordinates.
    Saves the generated heatmaps to the specified output directory.
    """
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    for slide_id, paths in dic_paths.items():
        try:
            # Ensure all required paths are present
            if len(paths) < 3:
                print(f"Skipping {slide_id}: Insufficient paths provided.")
                continue

            # Load patch coordinates
            with h5py.File(paths[2], 'r') as f:
                coords = f['coords'][:]

            # Initialize WholeSlideImage
            wsi = WholeSlideImage(paths[1])

            # Load scores
            scores = np.load(paths[0])

            # Generate heatmap and overlay
            heatmap, overlay, _ = wsi.visHeatmap(
                scores,
                coords,
                vis_level=5,
                segment=False,
                blank_canvas=False,
                alpha=0.4,
                patch_size=(cfg['patch_size'], cfg['patch_size']),
                cmap='jet',
                blur=cfg['blur'],
                convert_to_percentiles=cfg['percentiles']
            )

            # Save heatmap
            heatmap_path = output_dir / f"{slide_id}.jpeg"
            heatmap.save(str(heatmap_path))
            print(f"Heatmap saved: {heatmap_path}")

        except Exception as e:
            print(f"Error processing slide {slide_id}: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <config_file>")
        sys.exit(1)

    # Load configuration
    config_path = sys.argv[1]
    cfg = load_config(config_path)

    # Collect paths for slides, scores, and patches
    dic_paths = collect_paths(cfg)

    # Generate heatmaps
    generate_heatmaps(dic_paths, cfg)

    print("Processing finished!")


if __name__ == "__main__":
    main()