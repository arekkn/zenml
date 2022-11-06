from zenml.pipelines import pipeline

@pipeline
def embeddings_pipeline(
    save_names,
    save_resnet18_embeddings,
    save_resnet34_embeddings,
    save_resnet50_embeddings,
    save_resnet101_embeddings,
    save_resnet152_embeddings,
    merge_embeddings):
    filenames=save_names()
    embedding18=save_resnet18_embeddings(filenames)
    embedding34=save_resnet34_embeddings(filenames)
    embedding50=save_resnet50_embeddings(filenames)
    embedding101=save_resnet101_embeddings(filenames)
    embedding152=save_resnet152_embeddings(filenames)
    merge_embeddings(filenames, embedding18, embedding34, embedding50, embedding101, embedding152)

if __name__ == "__main__":
    from steps.embedding18 import save_resnet18_embeddings
    from steps.embedding34 import save_resnet34_embeddings
    from steps.embedding152 import save_resnet152_embeddings
    from steps.embedding50 import save_resnet50_embeddings
    from steps.embedding101 import save_resnet101_embeddings
    from steps.save_names import save_names
    from steps.save_embeddings import merge_embeddings
    embeddings_pipeline_instance = embeddings_pipeline(
        save_names(),
        save_resnet18_embeddings(),
        save_resnet34_embeddings(),
        save_resnet50_embeddings(),
        save_resnet101_embeddings(),
        save_resnet152_embeddings(),
        merge_embeddings()
    )

    embeddings_pipeline_instance.run()
    
