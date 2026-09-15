"""Tests for dataset loading and the notebook's pandas campaign pattern."""

import matplotlib
import pandas as pd

from dbtt import CharpySimulator, dataset
from dbtt.plotting import plot_dataset
from dbtt.simulator import COLUMNS

matplotlib.use("Agg")


def build_campaign(simulator, batches):
    """The pattern the notebook uses: run a batch, concat it on."""
    df = pd.DataFrame(columns=COLUMNS)
    for temperatures in batches:
        rows = simulator.run(temperatures, start_index=len(df) + 1)
        df = pd.concat([df, rows], ignore_index=True)
    return df


def test_campaign_accumulates_with_continuous_numbering():
    df = build_campaign(CharpySimulator(), [[-100, -40], [0, 40]])
    assert len(df) == 4
    assert list(df.columns) == COLUMNS
    assert df["Test house specimen ID"].tolist() == [1, 2, 3, 4]
    assert df["NEURONE specimen ID"].tolist()[-1] == "NRN2025-0004"
    assert "DBTT curve (J)" not in df.columns


def test_campaign_roundtrips_through_csv(tmp_path):
    df = build_campaign(CharpySimulator(), [[-100, -40, 0]])
    path = tmp_path / "campaign.csv"
    df.to_csv(path, index=False)

    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == COLUMNS
    pd.testing.assert_frame_equal(
        reloaded, df.astype(reloaded.dtypes.to_dict()), check_dtype=False
    )


def test_dataset_path_resolves_bare_names():
    assert dataset.dataset_path("my_campaign") == dataset.DATA_DIR / "my_campaign.csv"
    assert dataset.dataset_path("my_campaign.csv") == dataset.DATA_DIR / "my_campaign.csv"


def test_dataset_path_leaves_real_paths_alone(tmp_path):
    explicit = tmp_path / "elsewhere.csv"
    assert dataset.dataset_path(explicit) == explicit


def test_load_reads_a_csv(tmp_path):
    df = build_campaign(CharpySimulator(), [[-40]])
    path = tmp_path / "campaign.csv"
    df.to_csv(path, index=False)
    assert len(dataset.load(path)) == 1


def test_plot_dataset_accepts_a_dataframe():
    df = build_campaign(CharpySimulator(), [[-100, -60, -40, 0, 40]])
    ax = plot_dataset(df)
    assert ax.get_xlabel() == "Test temperature [C]"
    assert ax.collections[0].get_offsets().shape[0] == 5


def test_plot_dataset_accepts_a_path(tmp_path):
    df = build_campaign(CharpySimulator(), [[-40, 0]])
    path = tmp_path / "campaign.csv"
    df.to_csv(path, index=False)
    ax = plot_dataset(path)
    assert ax.collections[0].get_offsets().shape[0] == 2
