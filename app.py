from pathlib import Path

import streamlit as st

from utils import load_config


def parse_sample_label(path: Path) -> tuple[list[int], list[int]]:
    with open(path / "label.txt", "r") as file:
        lines = file.read().splitlines()
        init, final = [list(map(int, line.split(" "))) for line in lines]
        return init, final


DS_PATH = Path.cwd() / "dataset"
CONFIG_PATH = Path.cwd() / "strips.yml"

config = load_config(CONFIG_PATH)
item_mapping = {item.id: item.name for item in config.items}
item_mapping[0] = "blank"

st.title("NeurASP Crafting")

data_split = st.selectbox(label="select data split", options=["train", "valid", "test"])

split_path = DS_PATH / data_split
sample_paths = sorted(list(split_path.iterdir()), key=lambda p: int(p.stem))

st.write(f"Number of Samples: {len(sample_paths)}")

sample = st.selectbox(
    label="select sample", options=map(lambda p: p.stem, sample_paths)
)

with st.container(border=True):
    st.write("## Sample")
    sample_path = split_path / str(sample)
    st.write(f"Path: `{sample_path}`")

    init_ids, final_ids = parse_sample_label(sample_path)
    init = list(map(item_mapping.__getitem__, init_ids))
    final = list(map(item_mapping.__getitem__, final_ids))

    st.write("### Initial Inventory")
    init_paths = sorted(list(sample_path.glob("init_img_*.png")))
    init_cols = st.columns(len(init_paths))
    for idx, (path, col) in enumerate(zip(init_paths, init_cols)):
        col.image(path, width=128, caption=init[idx])

    st.write("### Final Inventory")
    final_paths = sorted(list(sample_path.glob("final_img_*.png")))
    final_cols = st.columns(len(final_paths))
    for idx, (path, col) in enumerate(zip(final_paths, final_cols)):
        col.image(path, width=128, caption=final[idx])
