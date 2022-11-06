from zenml.pipelines import pipeline

@pipeline
def embeddings_pipeline(
    save_names,
    save_resnet18_embeddings,
    merge_embeddings):
    filenames=save_names()
    embedding18=save_resnet18_embeddings(filenames)

    merge_embeddings(filenames, embedding18)

def run_2():
    from steps.embedding18 import save_resnet18_embeddings
    from steps.save_names import save_names
    from steps.save_embeddings import merge_embeddings
    embeddings_pipeline_instance = embeddings_pipeline(
        save_names(),
        save_resnet18_embeddings(),
        merge_embeddings()
    )
    embeddings_pipeline_instance.run()

# if __name__ == "__main__":
#     from steps.embedding18 import save_resnet18_embeddings
#     from steps.save_names import save_names
#     from steps.save_embeddings import merge_embeddings
#     embeddings_pipeline_instance = embeddings_pipeline(
#         save_names(),
#         save_resnet18_embeddings(),
#         merge_embeddings()
#     )

#     embeddings_pipeline_instance.run()