'''
ECAPA-TDNN run function 

Last modified: 05/2023
Author: Daniela Wiepert
Email: wiepert.daniela@mayo.edu
File: run.py
'''

#IMPORTS
#built-in
import argparse
import os
import pickle

#third-party
import numpy as np
import torch
import pandas as pd
import pyarrow

from google.cloud import storage
from torch.utils.data import  DataLoader

#local
from utilities import *
from models import *
from dataloader import AudioDataset
from loops import *

#main running functions
def train_timm(args):
    """
    Runtraining from start to finish
    :param args: dict with all the argument values
    """
    print('Running training: ')
    # (1) load data
    assert '.csv' not in args.data_split_root, f'May have given a full file path, please confirm this is a directory: {args.data_split_root}'
    train_df, val_df, test_df = load_data(args.data_split_root, args.target_labels, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)

    if args.debug:
        train_df = train_df.iloc[0:8,:]
        val_df = val_df.iloc[0:8,:]
        test_df = test_df.iloc[0:8,:]

    #(2) set audio configurations (val loader and eval loader will both use the eval_audio_conf
    train_audio_conf = {'dataset': args.dataset, 'mode': 'train', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}
    
    #note, mixup should always be 0 for the evaluation
    eval_audio_conf = {'dataset': args.dataset, 'mode': 'evaluation', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': 0, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}
    
    #(3) Generate audio dataset, note that if bucket not given, it assumes None and loads from local files
    train_dataset = AudioDataset(annotations_df=train_df, target_labels=args.target_labels, audio_conf=train_audio_conf, 
                                 prefix=args.prefix, bucket=args.bucket, librosa=args.lib) #librosa = True (might need to debug this one)
    val_dataset = AudioDataset(annotations_df=val_df, target_labels=args.target_labels, audio_conf=eval_audio_conf, 
                                 prefix=args.prefix, bucket=args.bucket, librosa=args.lib) #librosa = True (might need to debug this one)
    eval_dataset = AudioDataset(annotations_df=test_df, target_labels=args.target_labels, audio_conf=eval_audio_conf, 
                                prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    #(4) set up data loaders (val loader always has batchsize 1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # (4) initialize model
    model = timmForSpeechClassification(args.model_type, args.n_class, args.activation, args.final_dropout, args.layernorm)
    
    #(5) run model
    model = train(model, train_loader, val_loader, 
                  args.optim, args.learning_rate, args.weight_decay,
                  args.loss, args.scheduler, args.max_lr, args.epochs,
                  args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    
    print('Saving final model')
    mdl_path = os.path.join(args.exp_dir, '{}_{}_{}_epoch{}_{}_mdl.pt'.format(args.dataset, args.n_class, args.optim, args.epochs, args.model_type))
    torch.save(model.state_dict(), mdl_path)

    if args.cloud:
        upload(args.cloud_dir, mdl_path, args.bucket)

    # (6) start evaluating
    preds, targets = evaluation(model, eval_loader)
    
    print('Saving predictions and targets')
    pred_path = os.path.join(args.exp_dir, '{}_{}_{}_epoch{}_{}_predictions.pt'.format(args.dataset, args.n_class, args.optim, args.epochs, args.model_type))
    target_path = os.path.join(args.exp_dir, '{}_{}_{}_epoch{}_{}_targets.pt'.format(args.dataset, args.n_class, args.optim, args.epochs, args.model_type))
    torch.save(preds, pred_path)
    torch.save(targets, target_path)

    if args.cloud:
        upload(args.cloud_dir, pred_path, args.bucket)
        upload(args.cloud_dir, target_path, args.bucket)

    print('Training finished')


def eval_only(args):
    """
    Run only evaluation of a pre-existing model
    :param args: dict with all the argument values
    """
    assert args.trained_mdl_path is not None, 'must give a model to load'
    # get original model args (or if no finetuned model, uses your original args)
    model_args = load_args(args, args.trained_mdl_path)
    # (1) load data
    if '.csv' in args.data_split_root: 
        eval_df = pd.read_csv(args.data_split_root, index_col = 'uid')

        if 'distortions' in args.target_labels and 'distortions' not in eval_df.columns:
            eval_df["distortions"]=((eval_df["distorted Cs"]+eval_df["distorted V"])>0).astype(int)
        
        eval_df = eval_df.dropna(subset=args.target_labels)
    else:
        train_df, val_df, eval_df = load_data(args.data_split_root, args.target_labels, args.exp_dir, args.cloud, args.cloud_dir, args.bucket)
    
    if args.debug:
        eval_df = eval_df.iloc[0:8,:]

    #(2) set audio configurations (val loader and eval loader will both use the eval_audio_conf
    args.mixup=0 #mixup should always be 0 for evaluation only
    eval_audio_conf = {'dataset': args.dataset, 'mode': 'evaluation', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}
    

    # (3) set up datasets and dataloaders
    eval_dataset = AudioDataset(annotations_df=eval_df, target_labels=args.target_labels, audio_conf=eval_audio_conf, 
                                prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    
    #(4) set up data loaders (val loader always has batchsize 1)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # (4) initialize model
    model = timmForSpeechClassification(model_args.model_type, model_args.n_class, model_args.activation, model_args.final_dropout, model_args.layernorm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.trained_mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)

    # (6) start evaluating
    preds, targets = evaluation(model, eval_loader)

    print('Saving predictions and targets')
    pred_path = os.path.join(args.exp_dir, '{}_predictions.pt'.format(args.dataset))
    target_path = os.path.join(args.exp_dir, '{}_targets.pt'.format(args.dataset))
    torch.save(preds, pred_path)
    torch.save(targets, target_path)

    if args.cloud:
        upload(args.cloud_dir, pred_path, args.bucket)
        upload(args.cloud_dir, target_path, args.bucket)
    print('Evaluation finished')

def get_embeddings(args):
    """
    Run embedding extraction from start to finish
    :param args: dict with all the argument values
    """

    print('Running Embedding Extraction: ')
    assert args.trained_mdl_path is not None, 'must give a model to load for embedding extraction. '
    # Get original 
    model_args = load_args(args, args.trained_mdl_path)

    # (1) load data to get embeddings for
    assert '.csv' in args.data_split_root, f'A csv file is necessary for embedding extraction. Please make sure this is a full file path: {args.data_split_root}'
    annotations_df = pd.read_csv(args.data_split_root, index_col = 'uid') #data_split_root should use the CURRENT arguments regardless of the finetuned model

    if 'distortions' in args.target_labels and 'distortions' not in annotations_df.columns:
        annotations_df["distortions"]=((annotations_df["distorted Cs"]+annotations_df["distorted V"])>0).astype(int)

    if args.debug:
        annotations_df = annotations_df.iloc[0:8,:]

    #(2) set audio configurations
    args.mixup=0 #mixup should always be 0 for embedding extraction
    audio_conf = {'dataset': args.dataset, 'mode': 'evaluation', 'resample_rate': args.resample_rate, 'reduce': args.reduce, 'clip_length': args.clip_length,
                    'tshift':args.tshift, 'speed':args.speed, 'gauss_noise':args.gauss, 'pshift':args.pshift, 'pshiftn':args.pshiftn, 'gain':args.gain, 'stretch': args.stretch,
                    'num_mel_bins': args.num_mel_bins, 'target_length': args.target_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'noise':args.noise,
                    'mean':args.dataset_mean, 'std':args.dataset_std, 'skip_norm':args.skip_norm}

    # (3) set up dataloader with current args
    dataset = AudioDataset(annotations_df=annotations_df, target_labels=args.target_labels, audio_conf=audio_conf, 
                                prefix=args.prefix, bucket=args.bucket, librosa=args.lib)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn) 

    
    # (4) initialize model
    model = timmForSpeechClassification(model_args.model_type, model_args.n_class, model_args.activation, model_args.final_dropout, model_args.layernorm)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(args.trained_mdl_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    
    # (5) get embeddings
    embeddings = embedding_extraction(model, loader, args.embedding_type)
        
    df_embed = pd.DataFrame([[r] for r in embeddings], columns = ['embedding'], index=annotations_df.index)

    try:
        pqt_path = '{}/{}_{}_embeddings.pqt'.format(args.exp_dir, args.dataset, args.embedding_type)
        
        df_embed.to_parquet(path=pqt_path, index=True, engine='pyarrow') 

        if args.cloud:
            upload(args.cloud_dir, pqt_path, args.bucket)
    except:
        print('Unable to save as pqt, saving instead as csv')
        csv_path = '{}/{}_{}_embeddings.csv'.format(args.exp_dir, args.dataset, args.embedding_type)
        df_embed.to_csv(csv_path, index=True)

        if args.cloud:
            upload(args.cloud_dir, csv_path, args.bucket)

    print('Embedding extraction finished')
    return df_embed


def main():
    parser = argparse.ArgumentParser()
    #Inputs
    parser.add_argument('-i','--prefix',default='speech_ai/speech_lake/', help='Input directory or location in google cloud storage bucket containing files to load')
    parser.add_argument("-s", "--study", choices = ['r01_prelim','speech_poc_freeze_1', None], default='speech_poc_freeze_1', help="specify study name")
    parser.add_argument("-d", "--data_split_root", default='gs://ml-e107-phi-shared-aif-us-p/speech_ai/share/data_splits/amr_subject_dedup_594_train_100_test_binarized_v20220620/test.csv', help="specify file path where datasplit is located. If you give a full file path to classification, an error will be thrown. On the other hand, evaluation and embedding expects a single .csv file.")
    parser.add_argument('-l','--label_txt', default='src/labels.txt')
    parser.add_argument('--lib', default=False, type=bool, help="Specify whether to load using librosa as compared to torch audio")
    parser.add_argument("--trained_mdl_path", default=None, help="specify path to trained model")
    parser.add_argument("--model_type", default='efficientnet_b0', help='specify the timm model type to initialize')
    #GCS
    parser.add_argument('-b','--bucket_name', default='ml-e107-phi-shared-aif-us-p', help="google cloud storage bucket name")
    parser.add_argument('-p','--project_name', default='ml-mps-aif-afdgpet01-p-6827', help='google cloud platform project name')
    parser.add_argument('--cloud', default=False, type=bool, help="Specify whether to save everything to cloud")
    #output
    parser.add_argument("--dataset", default=None,type=str, help="When saving, the dataset arg is used to set file names. If you do not specify, it will assume the lowest directory from data_split_root")
    parser.add_argument("-o", "--exp_dir", default='./experiments', help='specify LOCAL output directory')
    parser.add_argument('--cloud_dir', default='', type=str, help="if saving to the cloud, you can specify a specific place to save to in the CLOUD bucket")
    #Mode specific
    parser.add_argument("-m", "--mode", choices=['train','eval','extraction'], default='train')
    parser.add_argument('--embedding_type', type=str, default='ft', help='specify whether embeddings should be extracted from classification head (ft) or base pretrained model (pt)', choices=['ft','pt'])
    #Audio configuration parameters
    parser.add_argument("--dataset_mean", default=-4.2677393, type=float, help="the dataset mean, used for input normalization")
    parser.add_argument("--dataset_std", default=4.5689974, type=float, help="the dataset std, used for input normalization")
    parser.add_argument("--target_length", default=1024, type=int, help="the input length in frames")
    parser.add_argument("--num_mel_bins", default=128,type=int, help="number of input mel bins")
    parser.add_argument("--resample_rate", default=16000,type=int, help='resample rate for audio files')
    parser.add_argument("--reduce", default=True, type=bool, help="Specify whether to reduce to monochannel")
    parser.add_argument("--clip_length", default=10.0, type=int, help="If truncating audio, specify clip length in seconds. 0 = no truncation")
    parser.add_argument("--tshift", default=0, type=float, help="Specify p for time shift transformation")
    parser.add_argument("--speed", default=0, type=float, help="Specify p for speed tuning")
    parser.add_argument("--gauss", default=0, type=float, help="Specify p for adding gaussian noise")
    parser.add_argument("--pshift", default=0, type=float, help="Specify p for pitch shifting")
    parser.add_argument("--pshiftn", default=0, type=float, help="Specify number of steps for pitch shifting")
    parser.add_argument("--gain", default=0, type=float, help="Specify p for gain")
    parser.add_argument("--stretch", default=0, type=float, help="Specify p for audio stretching")
    parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
    parser.add_argument('--timem', help='time mask max length', type=int, default=0)
    parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
    parser.add_argument("--noise", type=bool, default=False, help="specify if augment noise in finetuning")
    parser.add_argument("--skip_norm", type=bool, default=False, help="specify whether to skip normalization on spectrogram")
    #Model parameters
    parser.add_argument("-bs", "--batch_size", type=int, default=8, help="specify batch size")
    parser.add_argument("-nw", "--num_workers", type=int, default=0, help="specify number of parallel jobs to run for data loader")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0003, help="specify learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="specify number of training epochs")
    parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=.0001, help='specify weight decay for adamw')
    parser.add_argument("--loss", type=str, default="BCE", help="the loss function for finetuning, depend on the task", choices=["MSE", "BCE"])
    parser.add_argument("--scheduler", type=str, default="onecycle", help="specify lr scheduler", choices=["onecycle", None])
    parser.add_argument("--max_lr", type=float, default=0.01, help="specify max lr for lr scheduler")
    #classification head parameters
    parser.add_argument("--activation", type=str, default='relu', help="specify activation function to use for classification head")
    parser.add_argument("--final_dropout", type=float, default=0.3, help="specify dropout probability for final dropout layer in classification head")
    parser.add_argument("--layernorm", type=bool, default=False, help="specify whether to include the LayerNorm in classification head")
    #OTHER
    parser.add_argument("--debug", default=False, type=bool)
    args = parser.parse_args()
    
    print('Torch version: ',torch.__version__)
    print('Cuda availability: ', torch.cuda.is_available())
    print('Cuda version: ', torch.version.cuda)
    
    #variables
    # (1) Set up GCS
    if args.bucket_name is not None:
        storage_client = storage.Client(project=args.project_name)
        bucket = storage_client.bucket(args.bucket_name)
    else:
        bucket = None

    # (2), check if given study or if the prefix is the full prefix.
    if args.study is not None:
        args.prefix = os.path.join(args.prefix, args.study)
    
    # (3) get dataset name
    if args.dataset is None:
        if args.trained_mdl_path is None or args.mode == 'train':
            if '.csv' in args.data_split_root:
                args.dataset = '{}_{}'.format(os.path.basename(os.path.dirname(args.data_split_root)), os.path.basename(args.data_split_root[:-4]))
            else:
                args.dataset = os.path.basename(args.data_split_root)
        else:
            args.dataset = os.path.basename(args.trained_mdl_path)[:-7]
    
    # (4) get target labels
     #get list of target labels
    if args.label_txt is None:
        assert args.mode == 'extraction', 'Must give a txt with target labels for training or evaluating.'
        args.target_labels = None
        args.n_class = 0
    else:
        with open(args.label_txt) as f:
            target_labels = f.readlines()
        target_labels = [l.strip() for l in target_labels]
        args.target_labels = target_labels

        args.n_class = len(target_labels)

        if args.n_class == 0:
            assert args.mode == 'extraction', 'Target labels must be given for training or evaluating. Txt file was empty.'


    # (5) check if output directory exists, SHOULD NOT BE A GS:// path
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)

    # (6) check that clip length has been set
    if args.clip_length == 0:
        try: 
            assert args.batch_size == 1, 'Not currently compatible with different length wav files unless batch size has been set to 1'
        except:
            args.batch_size = 1
    
    # (7) dump arguments
    args_path = "%s/args.pkl" % args.exp_dir
    with open(args_path, "wb") as f:
        pickle.dump(args, f)
    #in case of error, everything is immediately uploaded to the bucket
    if args.cloud:
        upload(args.cloud_dir, args_path, bucket)

    # (8) check if trained model is stored in gcs bucket or confirm it exists on local machine
    if args.trained_mdl_path is not None:
        args.trained_mdl_path = gcs_model_exists(args.trained_mdl_path, args.bucket_name, args.exp_dir, bucket)

    #(9) add bucket to args
    args.bucket = bucket

    # (10) run model
    print(args.mode)
    if args.mode == "train":
        train_timm(args)

    elif args.mode == 'eval':
        eval_only(args)
              
    elif args.mode == "extraction":
        df_embed = get_embeddings(args)
    
if __name__ == "__main__":
    main()