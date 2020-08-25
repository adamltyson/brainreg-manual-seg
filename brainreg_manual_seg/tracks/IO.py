import pandas as pd


def save_track_layers(
    tracks_directory, points_layers, track_file_extension=".points",
):
    print(f"Saving tracks to: {tracks_directory}")
    convert_and_save_points(
        points_layers,
        tracks_directory,
        track_file_extension=track_file_extension,
    )


def convert_and_save_points(
    points_layers, output_directory, track_file_extension=".points",
):
    output_directory.mkdir(parents=True, exist_ok=True)

    for points_layer in points_layers:
        save_single_track_layer(
            points_layer,
            output_directory,
            track_file_extension=track_file_extension,
        )


def save_single_track_layer(
    layer, output_directory, track_file_extension=".points",
):
    output_filename = output_directory / (layer.name + track_file_extension)
    points = pd.DataFrame(layer.data)
    points.to_hdf(output_filename, key="df", mode="w")
