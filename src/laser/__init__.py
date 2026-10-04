import modal
from pandas import DataFrame

app = modal.App("Laser")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands("pip install 'pip<24.1'")
    .pip_install_from_pyproject("pyproject.toml")
    .run_commands("pip install --upgrade pip")
    .run_commands("pip uninstall fairseq -y")
    .apt_install("git")
    .run_commands(
        "pip install git+https://github.com/One-sixth/fairseq.git"
    )
)
@app.function(
    image=image,
    gpu="A10",
    timeout=60 * 60,
)
def run_model(text : str):
    import numpy as np
    import pandas as pd
    import laser_encoders
    from huggingface_hub import  create_repo , login
    from laser_encoders import LaserEncoderPipeline
    import matplotlib.pyplot as plt
    from datasets import Dataset, load_dataset
    token = ''
    repo_id = 'RanaGaber/Reddit_Egy_Block'

    ds = load_dataset(text)
    ds = ds["train"]
    df = ds.to_pandas()

    arabic_text = df["codafied_text_new"].to_list()
    english_text = df["english_translation"].to_list()

    encoder_ar = LaserEncoderPipeline(lang="arz_Arab")
    encoder_en = LaserEncoderPipeline(lang="eng_Latn")

    BATCH_SIZE = 512
    similarity_scores = []

    print(f"Starting batch encoding for {len(arabic_text)} sentences...")

    for i in range(0, len(arabic_text), BATCH_SIZE):
        batch_ar = arabic_text[i : i + BATCH_SIZE]
        batch_en = english_text[i : i + BATCH_SIZE]

        vec_ar = encoder_ar.encode_sentences(
            batch_ar,
            normalize_embeddings=True
        )

        vec_en = encoder_en.encode_sentences(
            batch_en,
            normalize_embeddings=True
        )

        batch_scores = np.sum(vec_ar * vec_en, axis=1)
        similarity_scores.extend(batch_scores.tolist())

        print(
            f"Processed rows {i} to "
            f"{min(i + BATCH_SIZE, len(arabic_text))}"
        )

    df["LASER"] = similarity_scores

    df.to_csv(
        "Laser_scores.csv",
        encoding="utf-8-sig",
        index=False
    )

    plt.hist(df["LASER"], bins=100)
    plt.title("LASER Histogram")
    plt.savefig(
        "./LASER.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    ds = Dataset.from_pandas(df)
    login(token = token)
    create_repo(
        repo_id = repo_id,
        token = token,
        repo_type= 'dataset',
        exist_ok = True
    )
    ds.push_to_hub(repo_id)
    return ds

@app.local_entrypoint()
def main():
    result = run_model.remote("SeragAmin/Egyptian_Reddit_Egy_en_block")
    print(result)