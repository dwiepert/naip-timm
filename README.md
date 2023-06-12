# TIMM for Mayo Data
This is a flexible implementation for training timm models from scratch. It is compatible with different timm model implementations. 

The command line usable, start-to-finish implementation for Mayo speech data is available with [run.py](https://github.com/dwiepert/mayo-timm/blob/main/src/run.py). A notebook tutorial version is also available at [run.ipynb](https://github.com/dwiepert/mayo-timm/blob/main/src/run.ipynb). 


## Running requirements
The environment must include the following packages, all of which can be dowloaded with pip or conda:
* albumentations
* librosa
* torch, torchvision, torchaudio
* tqdm (this is essentially enumerate(dataloader) except it prints out a nice progress bar for you)
* pyarrow
* timm

If running on your local machine and not in a GCP environment, you will also need to install:
* google-cloud-storage

The [requirements.txt](https://github.com/dwiepert/mayo-timm/blob/main/requirements.txt) can be used to set up this environment. 

To access data stored in GCS on your local machine, you will need to additionally run

```gcloud auth application-default login```

```gcloud auth application-defaul set-quota-project PROJECT_NAME```

Please note that if using GCS, the model expects arguments like model paths or directories to start with `gs://BUCKET_NAME/...` with the exception of defining an output cloud directory which should just be the prefix to save within a bucket. 

## Model type
In order to initialize a model you must specify a timm model architecture to prepare using `--model_type`. Here is a list of models is available here: [TIMM models](https://smp.readthedocs.io/en/latest/encoders_timm.html). We use `efficientnet_b0` as default. 

## Data structure
This code will only function with the following data structure.

SPEECH DATA DIR

    |

    -- UID 

        |

        -- waveform.EXT (extension can be any audio file extension)

        -- metadata.json (containing the key 'encoding' (with the extension in capital letters, i.e. mp3 as MP3), also containing the key 'sample_rate_hz' with the full sample rate)

and for the data splits

DATA SPLIT DIR

    |

    -- train.csv

    -- test.csv
    
## Audio Configuration
Data is loaded using an `AudioDataset` class, where you pass a dataframe of the file names (UIDs) along with columns containing label data, a list of the target labels (columns to select from the df), specify audio configuration, method of loading, and initialize transforms on the raw waveform and spectrogram (see [dataloader.py](https://github.com/dwiepert/mayo-timm/blob/main/src/dataloader.py)). You will need to access the fbank (input) and labels as follows: batch['fbank'], batch['targets]. 

To specify audio loading method, you can alter the `bucket` variable and `librosa` variable. As a default, `bucket` is set to None, which will force loading from the local machine. If using GCS, pass a fully initialized bucket. Setting the `librosa` value to 'True' will cause the audio to be loaded using librosa rather than torchaudio. 

The audio configuration parameters should be given as a dictionary (which can be seen in [run.py](https://github.com/dwiepert/mayo-timm/blob/main/src/runpy) and [run.ipynb](https://github.com/dwiepert/mayo-timm/blob/main/src/run.ipynb). Most configuration values are for initializing audio and spectrogram transforms. The transform will only be initialized if the value is not 0. If you have a further desire to add transforms, see [speech_utils.py](https://github.com/dwiepert/mayo-timm/blob/main/src/utilities/speech_utils.py)) and alter [dataloader.py](https://github.com/dwiepert/mayo-timm/blob/main/src/dataloader.py) accordingly. 

The following parameters are accepted (`--` indicates the command line argument to alter to set it):

*Dataset Information*
* `mean`: dataset mean (float). Set with `--dataset_mean`
* `std`: dataset standard deviation (float) Set with `--dataset_std`
*Audio Transform Information*
* `resample_rate`: an integer value for resampling. Set with `--resample_rate`
* `reduce`: a boolean indicating whether to reduce audio to monochannel. Set with `--reduce`
* `clip_length`: float specifying how many seconds the audio should be. Will work with the 'sample_rate' of the audio to get # of frames. Set with `--clip_length`
* `tshift`: Time shifting parameter (between 0 and 1). Set with `--tshift`
* `speed`: Speed tuning parameter (between 0 and 1). Set with `--speed`
* `gauss_noise`: amount of gaussian noise to add (between 0 and 1). Set with `--gauss`
* `pshift`: pitch shifting parameter (between 0 and 1). Set with `--pshift`
* `pshiftn`: number of steps for pitch shifting. Set with `--pshiftn`
* `gain`: gain parameter (between 0 and 1).Set with `--gain`
* `stretch`: audio stretching parameter (between 0 and 1). Set with `--stretch`
*Spectrogram Transform Information*
* `num_mel_bins`: number of frequency bins for converting from wav to spectrogram. Set with `--num_mel_bins`
* `target_length`: target length of resulting spectrogram. Set with `--target_length`
* `freqm`: frequency mask paramenter. Set with `--freqm`
* `timem`: time mask parameter. Set with `--timem`
* `noise`: add default noise to spectrogram. Set with `--noise`
* `mixup`: parameter for file mixup (between 0 and 1). Set with `--mixup`

Outside of the regular audio configurations, you can also set a boolean value for `cdo` (coarse drop out) and `shift` (affine shift) when initializing the `AudioDataset`. These are remnants of the original SSAST dataloading and not required. Both default to False. 

## Arguments
There are many possible arguments to set, including all the parameters associated with audio configuration. The main run function describes most of these, and you can alter defaults as required. 

### Loading data
* `-i, --prefix`: sets the `prefix` or input directory. Compatible with both local and GCS bucket directories containing audio files, though do not include 'gs://'
* `-s, --study`: optionally set the study. You can either include a full path to the study in the `prefix` arg or specify some parent directory in the `prefix` arg containing more than one study and further specify which study to select here.
* `-d, --data_split_root`: sets the `data_split_root` directory or a full path to a single csv file. For classification, it must be  a directory containing a train.csv and test.csv of file names. If runnning embedding extraction, it should be a csv file. Running evaluation only can accept either a directory or a csv file. This path should include 'gs://' if it is located in a bucket. 
* `-l, --label_txt`: sets the `label_txt` path. This is a full file path to a .txt file contain a list of the target labels for selection (see [labels.txt](https://github.com/dwiepert/mayo-ssast/blob/main/labels.txt))
* `--lib`: : specifies whether to load using librosa (True) or torchaudio (False), default=False
* `--trained_mdl_path`: specify a trained model if running evaluation only or extracting embeddings. This is a full file path to a pytorch model, and expects that whatever folder this is saved in includes an `args.pkl` file as well. 
* `--model_type`: specify the timm model type to initialize. Default is 'efficientnet_b0'

### Google cloud storage
* `-b, --bucket_name`: sets the `bucket_name` for GCS loading. Required if loading from cloud.
* `-p, --project_name`: sets the `project_name` for GCS loading. Required if loading from cloud. 
* `--cloud`: this specifies whether to save everything to GCS bucket. It is set as True as default.

### Saving data
* `--dataset`: Specify the name of the dataset you are using. When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root. Default is None. 
* `-o, --exp_dir`: sets the `exp_dir`, the LOCAL directory to save all outputs to. 
* `--cloud_dir`: if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket. Do not include the bucket_name or 'gs://' in this path.

### Run mode
* `-m, --mode`: Specify the mode you are running, i.e., whether to run fine-tuning for classification ('finetune'), evaluation only ('eval-only'), or embedding extraction ('extraction'). Default is 'finetune'.
* `--embedding_type`: specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)

### Audio transforms
see the audio configurations section for which arguments to set

### Training parameters
* `--batch_size`: set the batch size (default 8)
* `--num_workers`: set number of workers for dataloader (default 0)
* `--learning_rate`: you can manually change the learning rate (default 0.0003)
* `--epochs`: set number of training epochs (default 1)
* `--optim`: specify the training optimizer. Default is `adam`.
* `--weight_decay`: specify weight decay for AdamW optimizer
* `--loss`: specify the loss function. Can be 'BCE' or 'MSE'. Default is 'BCE'.
* `--scheduler`: specify a lr scheduler. If None, no lr scheduler will be use. The only scheduler option is 'onecycle', which initializes `torch.optim.lr_scheduler.OneCycleLR`
* `--max_lr`: specify the max learning rate for an lr scheduler. Default is 0.01.

### Classification Head parameters
* `--activation`: specify activation function to use for classification head
* `--final_dropout`: specify dropout probability for final dropout layer in classification head
* `--layernorm`: specify whether to include the LayerNorm in classification head

For more information on arguments, you can also run `python run.py -h`. 

## Functionality
This implementation contains many functionality options as listed below:

### 1. Training from scratch
You can train a timm model from scratch for classifying speech features using the `timmForSpeechClassification` class in [timm_models.py](https://github.com/dwiepert/mayo-timm/blob/main/src/models/timm_models.py) and the `train(...)` function in [loops.py](https://github.com/dwiepert/mayo-timm/blob/main/src/loops.py).

This mode is triggered by setting `-m, --mode` to 'train'. 

The classification head can be altered to use a different amount of dropout and to include/exclude layernorm. See `ClassificationHead` class in [speech_utils.py](https://github.com/dwiepert/mayo-timm/blob/main/src/utilities/speech_utils.py) for more information. 

Additionally, there are data augmentation transforms available for finetuning, such as time shift, speed tuning, adding noise, pitch shift, gain, stretching audio, and audio mixup. 

### 2. Evaluation only
If you have a trained model and want to evaluate it on a new data set, you can do so by setting `-m, --mode` to 'eval'. You must then also specify a `--trained_mdl_path` to load in. 

It is expected that there is an `args.pkl` file in the same directory as the model to indicate which arguments were used to initialize the model. This implementation will load the arguments and initialize/load the  model with these arguments. If no such file exists, it will use the arguments from the current run, which could be incompatible if you are not careful. 

### 3. Embedding extraction
We implemented multiple embedding extraction methods for use with the SSAST model. The implementation is a function within `timmForSpeechClassification` called `extract_embedding(x, embedding_type)`, which is called on batches instead of the forward function. 

Embedding extraction is triggered by setting `-m, --mode` to 'extraction'. 

You must also consider where you want the embeddings to be extracted from. The options are as follows:
1. From the output of the base ECAPA-TDNN model? Set `embedding_type` to 'pt'. 
2. From a layer in the classification head? Set `embedding_type` to 'ft'. This version will always return the output from the first dense layer in the classification head, prior to any activation function or normalization. 
