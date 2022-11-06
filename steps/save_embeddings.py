from zenml.steps import step
import pandas as pd

@step(enable_cache=False)
def merge_embeddings(names: list, resnet18_embeddings: list, resnet34_embeddings: list, resnet50_embeddings: list, resnet101_embeddings: list, resnet152_embeddings: list) -> None:
    result = pd.DataFrame({
        "img_paths":names,
        "resnet18_embeddings":resnet18_embeddings,
        "resnet34_embeddings":resnet34_embeddings,
        "resnet50_embeddings":resnet50_embeddings,
        "resnet101_embeddings":resnet101_embeddings,
        "resnet152_embeddings":resnet152_embeddings})
    result.to_pickle("./embeddings.pkl", compression="xz")