from huggingface_hub import upload_folder

upload_folder(
    repo_id="GetchEbuy/Demo_deeplearning",
    folder_path="model/gpt_smiles",
    repo_type="model"
)
