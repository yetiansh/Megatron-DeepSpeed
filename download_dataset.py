# Modifiy from https://github.com/openai/gpt-2-output-dataset/blob/master/detector/download.py
import os

import requests
from tqdm import tqdm

ALL_DATASETS = ["PhilPapersDataset_text_document", "EuroParlDataset_text_document"]


def download(*datasets, data_dir="/Megatron-DeepSpeed/data"):
    os.makedirs(data_dir, exist_ok=True)

    for ds in datasets:
        assert ds in ALL_DATASETS, f'Unknown dataset {ds}'

        for split in ["bin", "idx"]:
            filename = ds + "." + split
            output_file = os.path.join(data_dir, filename)
            if os.path.isfile(output_file):
                continue

            r = requests.get("https://mystic.the-eye.eu/public/AI/pile_neox/data/" + filename, stream=True)

            with open(output_file, "wb") as f:
                file_size = int(r.headers["content-length"])
                chunk_size = 1000
                with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                    # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(chunk_size)


if __name__ == '__main__':
    download(*ALL_DATASETS)
