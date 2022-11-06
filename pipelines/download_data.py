from zenml.pipelines import pipeline

@pipeline(enable_cache=False)
def download_and_preproccess_data(download_step, format_step, validate_step):
    format_step.after(download_step)
    validate_step.after(format_step)
    download_step()
    format_step()
    validate_step()



if __name__=='__main__':
    from steps.download_photos import download_cat_photos
    from steps.assert_all_photos_are_cats import check_cats
    from steps.crop_photos import format_photos
    pipeline = download_and_preproccess_data(
        download_cat_photos(),
        format_photos(),
        check_cats()
    )
    pipeline.run()