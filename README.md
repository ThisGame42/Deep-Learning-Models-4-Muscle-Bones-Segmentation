# Deep-Learning-Models-4-Muscle-Bones-Segmentation


This repo contains code for our NMR in Biomedicine paper - [Deep learning methods for automatic segmentation of lower leg muscles and bones from MRI scans of children with and without cerebral palsy](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/nbm.4609), which investigated and applied six deep learning models to segment individual lower leg muscles and bones from MRI scans of children with and without cerebral palsy.


The investigated models were: UNet, UNet++, 3D UNet, VNet, H-DenseUNet, and HybridUNet which is an in-house model inspired by H-DenseUNet. To learn more about UNet check out [here](https://arxiv.org/abs/1505.04597). The paper where UNet++ was proposed can be found [here](https://arxiv.org/abs/1912.05074), 3D UNet and VNet were proposed in [3D UNet](https://arxiv.org/abs/1606.06650) and in [VNet](https://arxiv.org/abs/1606.04797), Finally, H-DenseUNet was proposed in [this paper](https://arxiv.org/abs/1709.07330).

To train a model, run the following command in the root directory (of this repo):

```python
python3 ./entry.py --training_img_path [PATH_TO_YOUR_DATA] 
--training_label_path [PATH_TO_YOUR_LABELS] 
--model_name [NAME_OF_THE_MODEL] --val_img_path [PATH_TO_VAL_DATA]
--val_label_path [PATH_TO_VAL_LABELS] 
--loss_fn [LOSS_FN_NAME] --optimiser [OPTIMISER_OF_CHOOSING] --num_epochs [NUM_EPOCHS 
--output_path [YOUR_OUTPUT_PATH]  --batch_size [BATCH_SIZE] --num_classes [NAME_OF_CLASSES] --learning_rate [LR_RATE]
--pre_training [TRUE/FALSE] --is_2d [TRUE/FALSE]
```

Note that ```--is_2d``` needs to be ```True``` for 2D models and ```False``` for 3D models. Also note that the training data are expected to be NIFTI files. You need to modify data set classes if you wish to load data of other formats.


