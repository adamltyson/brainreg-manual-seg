from brainreg_manual_seg.man_seg_tools import analyse_track_anatomy
from brainreg_manual_seg.tracks.fit import spline_fit


def track_analysis(
    viewer,
    atlas,
    tracks_directory,
    track_layers,
    napari_spline_size,
    spline_points=100,
    fit_degree=3,
    spline_smoothing=0.05,
    summarise_track=True,
):
    tracks_directory.mkdir(parents=True, exist_ok=True)

    print(
        f"Fitting splines with {spline_points} segments, of degree "
        f"'{fit_degree}' to the points"
    )
    splines = []

    for track_layer in track_layers:
        spline = spline_fit(
            track_layer.data,
            smoothing=spline_smoothing,
            k=fit_degree,
            n_points=spline_points,
        )
        splines.append(spline)
        if summarise_track:
            summary_csv_file = tracks_directory / (track_layer.name + ".csv")
            analyse_track_anatomy(atlas, spline, summary_csv_file)

        viewer.add_points(
            spline,
            size=napari_spline_size,
            edge_color="cyan",
            face_color="cyan",
            blending="additive",
            opacity=0.7,
            name=track_layer.name + "_fit",
        )

    return splines
