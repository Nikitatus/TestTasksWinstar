def flatten_images(images_batch):
    batch_size = images_batch.shape[0]
    flattened_images = images_batch.reshape(batch_size, -1)
    return flattened_images