# SimSiam-TF

This is an unofficial implementation of SimSiam ([Exploring Simple Siamese Representation Learning, 2020.](https://arxiv.org/abs/2011.10566)).

## Requirements
- python >= 3.6
- tensorflow >= 2.2

## Training
To train SimSiam,
```
python main.py \
    --task pretext \
    --stop_gradient \
    --proj_bn_hidden \
    --proj_bn_output \
    --pred_bn_hidden \
    --weight_decay 0.0001 \
    --batch_size 256 \
    --epochs 200 \
    --lr_mode cosine \
    --data_path /path/of/your/data \
    --gpus gpu id(s) which will be used
```

## Evaluation
To evaluate pre-trained model with linear classification,
```
python main.py \
    --task lincls \
    --batch_size 256 \
    --epochs 90 \
    --lr 30 \
    --lr_mode cosine \
    --data_path /path/of/your/data \
    --snapshot /path/of/checkpoint \
    --gpus gpu id(s) which will be used
```

## Results
### ImageNet
|         Model         | batch | Accuracy (paper) | Accuracy (ours) |
| --------------------- | ----- | ---------------- | --------------- |
| ResNet50 (200 epochs) |  256  |       68.1       |       -         |

## Citation
```
@article{Chen2020ExploringSS,
  title={Exploring Simple Siamese Representation Learning},
  author={Xinlei Chen and Kaiming He},
  journal={ArXiv},
  year={2020},
  volume={abs/2011.10566}
}
```
