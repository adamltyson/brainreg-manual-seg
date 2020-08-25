from pathlib import Path
from brainreg_manual_seg.tracks.IO import brainrender_track_to_napari


def add_new_track_layer(viewer, track_layers, point_size):
    num = len(track_layers)
    new_track_layers = viewer.add_points(
        n_dimensional=True, size=point_size, name=f"track_{num}",
    )
    new_track_layers.mode = "ADD"
    track_layers.append(new_track_layers)


def add_existing_track_layers(viewer, track_file, point_size):
    max_z = len(viewer.layers[0].data)
    data = brainrender_track_to_napari(track_file, max_z)
    new_points_layer = viewer.add_points(
        data, n_dimensional=True, size=point_size, name=Path(track_file).stem,
    )
    new_points_layer.mode = "ADD"
    return new_points_layer
