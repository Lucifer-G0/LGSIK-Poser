# LGSIK-Poser
##  Requirements
- Python >= 3.9
- PyTorch >= 2.0.1
- numpy >= 1.23.1
- [human_body_prior](https://github.com/nghorbani/human_body_prior)


## :hammer_and_pick: Preparation

### AMASS

1. Please download the datasets from [AMASS](https://amass.is.tue.mpg.de/) and place them in `./data/AMASS` directory of this repository.
2. Download the required body models and place them in `./body_models` directory of this repository. For the SMPL+H body model, download it from http://mano.is.tue.mpg.de/. Please download the AMASS version of the model with DMPL blendshapes. You can obtain dynamic shape blendshapes, e.g. DMPLs, from http://smpl.is.tue.mpg.de.
3. clone https://github.com/nghorbani/human_body_prior/tree/master under this project directory
4. Run  `./prepare_data.py` to preprocess the input data for faster training. The data split for training and testing data under Protocol 1 in our paper is stored under the folder `./prepare_data/data_split` (directly copy from [AvatarPoser](https://github.com/eth-siplab/AvatarPoser)).

```
python ./prepare_data.py --support_dir ./body_models/ --root_dir ./data/AMASS/ --save_dir [path_to_save]
```
## :bicyclist: Training

Modify the dataset_path in `./options/train_config.yaml` to your `[path_to_save]`.

```
python train.py --config ./options/train_config.yaml
```

## :running_woman: Testing

1. Download and extract the pre-trained model from the [Release](https://github.com/Lucifer-G0/LGSIK-Poser/releases/tag/pretrained).
2. Modify config "options/test_config.yaml", espcially change the "resume_model" to the path of pretrained model
3. run files in test

   model_param.py: test model param、flops、inference speed

   test_class_real.py: sling window for online running

   test_class_weight.py: sequence length weighted offline running

   others mention in paper
