#!/usr/bin/env python3
"""
Generate the European sentiment geomap starting from the aggregated scores CSV.

This script recreates the visualization prepared for the hackathon, making it
easy to reproduce the final figure without opening the notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "Geomap" / "data_geomap_total.csv"
DEFAULT_SHAPE_PATH = PROJECT_ROOT / "map" / "ne_110m_admin_0_countries.shp"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "european_sentiment_map.png"


def load_scores_dataset(path: Path) -> pd.DataFrame:
    """Load and clean the aggregated review scores dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Scores dataset not found: {path}")

    df = pd.read_csv(path)
    required_columns = {"City_Nation", "Reviewer_Score"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"Dataset at {path} is missing columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["City_Nation", "Reviewer_Score"]).copy()
    df["Country"] = (
        df["City_Nation"]
        .astype(str)
        .str.split(",")
        .str[-1]
        .str.strip()
    )
    country_scores = (
        df.groupby("Country", as_index=False)["Reviewer_Score"]
        .mean()
        .rename(columns={"Country": "NAME"})
    )
    return country_scores


def load_europe_shapefile(path: Path) -> gpd.GeoDataFrame:
    """Load the Natural Earth shapefile and keep the European subset."""
    if not path.exists():
        raise FileNotFoundError(f"Shapefile not found: {path}")

    world = gpd.read_file(path)
    europe = world[(world["CONTINENT"] == "Europe") & (world["NAME"] != "Russia")]
    return europe


def plot_geomap(
    europe_gdf: gpd.GeoDataFrame,
    scores_df: pd.DataFrame,
    output_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    show: bool = False,
) -> None:
    """Merge spatial and tabular data, then render and save the figure."""
    merged = europe_gdf.merge(scores_df, on="NAME", how="left")

    cmap = LinearSegmentedColormap.from_list("white_to_blue", ["white", "#0b3d91"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    merged.plot(
        column="Reviewer_Score",
        cmap=cmap,
        linewidth=0.8,
        ax=ax,
        edgecolor="0.8",
        vmin=vmin,
        vmax=vmax,
        missing_kwds={"color": "#f0f0f0", "label": "No data"},
    )

    ax.set_title(title, fontsize=16)
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # type: ignore[attr-defined]
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.036, pad=0.04)
    cbar.set_label("Reviewer Score", fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the European sentiment geomap.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the aggregated review scores CSV.",
    )
    parser.add_argument(
        "--shape-path",
        type=Path,
        default=DEFAULT_SHAPE_PATH,
        help="Path to the Natural Earth shapefile (.shp).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="File path where the map image will be saved.",
    )
    parser.add_argument(
        "--title",
        default="Average Reviewer Score by Country",
        help="Title displayed on the map.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Minimum value for the color scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=10.0,
        help="Maximum value for the color scale.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the map window after saving the image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores_df = load_scores_dataset(args.data_path)
    europe_gdf = load_europe_shapefile(args.shape_path)
    plot_geomap(
        europe_gdf,
        scores_df,
        args.output,
        args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        show=args.show,
    )
    print(f"Geomap saved to {args.output}")


if __name__ == "__main__":
    main()
