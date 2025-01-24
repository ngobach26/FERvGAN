## Downloading datasets
To download the CelebA dataset:
https://www.kaggle.com/datasets/msambare/fer2013
https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset
generated dataset:
https://www.kaggle.com/datasets/bchngo/generated-fer

## Training StarGAN networks

```bash
# Train StarGAN on custom datasets
python main.py --mode train --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TRAIN_IMG_DIR \
               --sample_dir DIR_NAME/samples --log_dir DIR_NAME/logs \
               --model_save_dir DIR_NAME/models --result_dir DIR_NAME/results

# Test StarGAN on custom datasets
python main.py --mode test --rafd_crop_size CROP_SIZE --image_size IMG_SIZE \
               --c_dim LABEL_DIM --rafd_image_dir TEST_IMG_DIR \
               --sample_dir DIR_NAME/samples --log_dir DIR_NAME/logs \
               --model_save_dir DIR_NAME/models --result_dir DIR_NAME/results
```
## Training classification models


## Citation
[StarGAN](https://arxiv.org/abs/1711.09020)