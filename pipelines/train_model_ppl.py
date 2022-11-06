from zenml.pipelines import pipeline

@pipeline(enable_cache=False)
def train_model_ppl(load_model, train_model):
    a = load_model()
    train_model(a)


if __name__ == "__main__":
    from steps.load_model import load_model

    from steps.train_model import train_model

    pipeline = train_model_ppl(load_model(), train_model())
    pipeline.run()
