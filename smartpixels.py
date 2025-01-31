import torch.nn as nn
import torch
from activations_quantizer import QuantizedTanh, hard_tanh
import torch.nn.functional as F

class SmartPixelModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sep_conv = depthwise_separable_conv(input_channels=20, output_channels=5)
        self.conv = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=1)
        self.avg = nn.AvgPool2d(3)
        self.flat = nn.Flatten(1)
        self.dense1 = nn.Linear(90, 16)
        self.dense2 = nn.Linear(16,16)
        self.dense3 = nn.Linear(16, 14)

    def forward(self, x):
        x = self.sep_conv(x)
        x = F.tanh(x)
        x = self.conv(x)
        x = F.tanh(x)
        
        x = self.avg(x)
        x = F.tanh(x)
    
        x = self.flat(x)
        x = self.dense1(x)
        x = F.tanh(x)
        x = self.dense2(x)
        x = F.tanh(x)
        x = self.dense3(x)
        return x

class depthwise_separable_conv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=0, groups=input_channels, bias=False)
        self.pointwise = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# python imports
import tensorflow as tf
from typing import Union, List, Tuple
import glob
import numpy as np
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import logging
import gc
import torch
import shutil
from pathlib import Path


def safe_remove_directory(directory_path):
    if Path(directory_path).exists():
        print(f"Directory {directory_path} is removed...")
        shutil.rmtree(directory_path)
    else:
        print(f"Directory {directory_path} does not exist and cannot be removed.")

class OptimizedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
            data_directory_path: str = "./",
            labels_directory_path: str = "./",
            is_directory_recursive: bool = False,
            file_type: str = "csv",
            data_format: str = "2D",
            batch_size: int = 32,
            file_count = None,
            labels_list: Union[List,str] = "cotAlpha",
            to_standardize: bool = False,
            input_shape: Tuple = (13,21),
            transpose = None,
            include_y_local: bool = False,
            files_from_end = False,
            shuffle=False,
            current=False,
            sample_delta_t=200,

            # Added in Optimized datagenerators 
            load_from_tfrecords_dir: str = None,
            tfrecords_dir: str = None,
            use_time_stamps = -1,
            seed: int = None,
            quantize: bool = False,
            max_workers: int = 1,
                 
            **kwargs,
            ):
        super().__init__() 

        """
        Data Generator to streamline data input to the network direct from the directory.
        Args:
        data_directory_path:
        labels_directory_path: 
        is_directory_recursive: 
        file_type: Default: "csv"
                   Adapt the data loader according to file type. For now, it only supports csv and parquet file formats.
        data_format: Default: 2D
                     Used to refer to the relevant "recon" files, 2D for 2D pixel array, 3D for time series input,
        batch_size: Default: 32
                    The no. of data points to be included in a single batch.
        file_count: Default: None
                    To limit the no. of .csv files to be used for training.
                    If set to None, all files will be considered as legitimate inputs.
        labels_list: Default: "cotAlpha"
                     Input column name or list of column names to be used as label input to the neural network.
        to_standardize: If set to True, it ensures that batches are normalized prior to being used as inputs
                        for training.
                        Default: False
        input_shape: Default: (13,21) for image input to a 2D feedforward neural network.
                    To reshape the input array per the requirements of the network training.
        current: Default False, calculate the current instead of the integrated charge
        sample_delta_t: how long an "ADC bin" is in picoseconds
        
        load_from_tfrecords_dir: Directory to load prepared data from TFRecords.
        tfrecords_dir: Directory to save TFRecords.
        use_time_stamps: which of the 20 time stamps to train on. default -1 is to train on all of them
        seed: Random seed for shuffling.
        quantize: Whether to quantize the data.
        """


        # decide on which time stamps to load
        self.use_time_stamps = np.arange(0,20) if use_time_stamps == -1 else use_time_stamps
        len_xy, ntime = 13*21, 20
        idx = [[i*(len_xy),(i+1)*(len_xy)] for i in range(ntime)] # 20 time stamps of length 13*21
        self.use_time_stamps = np.array([ np.arange(idx[i][0], idx[i][1]).astype("str") for i in self.use_time_stamps]).flatten().tolist()
        if use_time_stamps != -1 and data_format != '2D':
            assert len(use_time_stamps) == input_shape[0]

        self.max_workers = max_workers
        self.shuffle = shuffle
        if shuffle:
            self.seed = seed if seed is not None else 13
            self.rng = np.random.default_rng(seed = self.seed)
        
        if file_type not in ["csv", "parquet"]:
            raise ValueError("file_type can only be \"csv\" or \"parquet\"!")
        self.file_type = file_type
        self.recon_files = glob.glob(
            data_directory_path + "recon" + data_format + "*." + file_type, 
            recursive=is_directory_recursive
        )
        
        self.recon_files.sort()
        if file_count != None:
            if not files_from_end:
                self.recon_files = self.recon_files[:file_count]
            else:
                self.recon_files = self.recon_files[-file_count:]

        self.label_files = [
            labels_directory_path + recon_file.split('/')[-1].replace("recon" + data_format, "labels") for recon_file in self.recon_files
        ]

        self.file_offsets = [0]
        self.dataset_mean = None
        self.dataset_std = None

        # If data is already prepared load anduse that data
        if load_from_tfrecords_dir is not None:
            if not os.path.isdir(load_from_tfrecords_dir):
                raise ValueError(f"Directory {load_from_tfrecords_dir} does not exist.")
            else:
                self.tfrecords_dir = load_from_tfrecords_dir
        else:
            safe_remove_directory(tfrecords_dir)
            self.batch_size = batch_size
            self.labels_list = labels_list
            self.input_shape = input_shape
            self.transpose = transpose
            self.to_standardize = to_standardize
            self.include_y_local = include_y_local

            self.process_file_parallel()
    
            self.current_file_index = None
            self.current_dataframes = None
    
            if tfrecords_dir is None:
                raise ValueError(f"tfrecords_dir is None")
                
            self.tfrecords_dir = tfrecords_dir    
            os.makedirs(self.tfrecords_dir, exist_ok=True)
            self.save_batches_parallel() # save all the batches
            del self.current_dataframes 
            
            
        self.tfrecord_filenames = np.sort(np.array(tf.io.gfile.glob(os.path.join(self.tfrecords_dir, "*.tfrecord"))))
        self.quantize = quantize
        self.epoch_count = 0
        self.on_epoch_end()

    def process_file_parallel(self):
        file_infos = [(afile, self.use_time_stamps, self.file_type, self.input_shape, self.transpose) for afile in self.recon_files]
        results = []
        with ProcessPoolExecutor(self.max_workers) as executor:
            futures = [executor.submit(self._process_file_single, file_info) for file_info in file_infos]
            for future in tqdm(as_completed(futures), total=len(file_infos), desc="Processing Files..."):
                results.append(future.result())

        for amean, avariance, amin, amax, num_rows in results:
            self.file_offsets.append(self.file_offsets[-1] + num_rows)

            if self.dataset_mean is None:
                self.dataset_max = amax
                self.dataset_min = amin
                self.dataset_mean = amean
                self.dataset_std = avariance
            else:
                self.dataset_max = max(self.dataset_max, amax)
                self.dataset_min = min(self.dataset_min, amin)
                self.dataset_mean += amean
                self.dataset_std += avariance

        self.dataset_mean = self.dataset_mean / len(self.recon_files)
        self.dataset_std = np.sqrt(self.dataset_std / len(self.recon_files)) 
            
        self.file_offsets = np.array(self.file_offsets)

    @staticmethod
    def _process_file_single(file_info):
        afile, use_time_stamps, file_type, input_shape, transpose = file_info
        if file_type == "csv":
            adf = pd.read_csv(afile).dropna()
        elif file_type == "parquet":
            adf = pd.read_parquet(afile, columns=use_time_stamps).dropna()
    
        x = adf.values
        nonzeros = abs(x) > 0
        x[nonzeros] = np.sign(x[nonzeros]) * np.log1p(abs(x[nonzeros])) / math.log(2)
        amean, avariance = np.mean(x[nonzeros], keepdims=True), np.var(x[nonzeros], keepdims=True) + 1e-10
        centered = np.zeros_like(x)
        centered[nonzeros] = (x[nonzeros] - amean) / np.sqrt(avariance)
        x = x.reshape((-1, *input_shape))
        if transpose is not None:
            x = x.transpose(transpose)
        amin, amax = np.min(centered), np.max(centered)
        len_adf = len(adf)
        del adf
        gc.collect()
        
        return amean, avariance, amin, amax, len_adf

    def standardize(self, x, norm_factor_pos=1.7, norm_factor_neg=2.5):
        """
        Applies the normalization configuration in-place to a batch of inputs.
        `x` is changed in-place since the function is mainly used internally
        to standardize images and feed them to your network.
        Args:
            x: Batch of inputs to be normalized.
        Returns:
            The inputs, normalized. 
        """
        out = (x - self.dataset_mean)/self.dataset_std
        out[out > 0] = out[out > 0]/norm_factor_pos
        out[out < 0] = out[out < 0]/norm_factor_neg
        return out

    def save_batches_parallel(self):
        """
        Saves data batches as multiple TFRecord files.
        """
        # TODO: Make this parallelized
        num_batches = self.__len__() # Total num of batches
        paths_or_errors = []

        # The max_workers is set to 1 because processing large batches in multiple threads can significantly
        # increase RAM usage. Adjust 'max_workers' based on your system's RAM capacity and requirements.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_batch = {executor.submit(self.save_single_batch, i): i for i in range(num_batches)}
            
            for future in tqdm(as_completed(future_to_batch), total=num_batches, desc="Saving batches as TFRecords"):
                result = future.result()
                paths_or_errors.append(result)
            
        for res in paths_or_errors:
            if "Error" in res:
                print(res)  
                
    def save_single_batch(self, batch_index):
        """
        Serializes and saves a single batch to a TFRecord file.
        Args:
            batch_index (int): Index of the batch to save.
        Returns:
            str: Path to the saved TFRecord file or an error message.
        """
        
        try:
            filename = f"batch_{batch_index}.tfrecord"
            TFRfile_path = os.path.join(self.tfrecords_dir, filename)
            X, y = self.prepare_batch_data(batch_index)
            serialized_example = self.serialize_example(X, y)
            with tf.io.TFRecordWriter(TFRfile_path) as writer:
                writer.write(serialized_example)
            return TFRfile_path
        except Exception as e:
            return f"Error saving batch {batch_index}: {e}" 
    
    def prepare_batch_data(self, batch_index):
        """
        Used to fetch a batch of inputs (X,y) for the network's training.
        """
        index = batch_index * self.batch_size # absolute *event* index
        
        file_index = np.arange(self.file_offsets.size)[index < self.file_offsets][0] - 1 # first index is 0!

        index = index - self.file_offsets[file_index] # relative event index in file
        batch_size = min(index + self.batch_size, self.file_offsets[file_index + 1] - self.file_offsets[file_index])
        
        if file_index != self.current_file_index:
            self.current_file_index = file_index
            # print()
            # print(self.recon_files[file_index])
            if self.file_type == "csv":
                recon_df = pd.read_csv(self.recon_files[file_index])
                labels_df = pd.read_csv(self.label_files[file_index])[self.labels_list]
            elif self.file_type == "parquet":
                recon_df = pd.read_parquet(self.recon_files[file_index], columns=self.use_time_stamps)
                labels_df = pd.read_parquet(self.label_files[file_index], columns=self.labels_list)

            has_nans = np.any(np.isnan(recon_df.values), axis=1)
            has_nans = np.arange(recon_df.shape[0])[has_nans]
            recon_df_raw = recon_df.drop(has_nans)
            labels_df_raw = labels_df.drop(has_nans)

            joined_df = recon_df_raw.join(labels_df_raw)

            if self.shuffle: # Changed
                joined_df = joined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)  

            recon_values = joined_df[recon_df_raw.columns].values            

            nonzeros = abs(recon_values) > 0
            
            recon_values[nonzeros] = np.sign(recon_values[nonzeros])*np.log1p(abs(recon_values[nonzeros]))/math.log(2)
            
            if self.to_standardize:
                recon_values[nonzeros] = self.standardize(recon_values[nonzeros])
            
            recon_values = recon_values.reshape((-1, *self.input_shape))            
                        
            if self.transpose is not None:
                recon_values = recon_values.transpose(self.transpose)
            
            self.current_dataframes = (
                recon_values, 
                joined_df[labels_df_raw.columns].values,
            )        
        
        recon_df, labels_df = self.current_dataframes

        # print(f'start_index: {index}\t end_index: {batch_size}')
        X = recon_df[index:batch_size]
        y = labels_df[index:batch_size] / np.array([75., 18.75, 8.0, 0.5])
    
        if self.include_y_local:
            y_local = labels_df.iloc[chosen_idxs]["y-local"].values
            return [X, y_local], y
        else:
            return X, y

    
    def serialize_example(self, X, y):
        """
        Serializes a single example (featuresand labels) to TFRecord format. 
        
        Args:
        - X: Training data
        - y: labelled data
        
        Returns:
        - string (serialized TFRecord example).
        """
        # X and y are float32 (maybe we can reduce this)
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)

        feature = {
            'X': self._bytes_feature(tf.io.serialize_tensor(X)),
            'y': self._bytes_feature(tf.io.serialize_tensor(y)),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """
        Converts a string/byte value into a Tf feature of bytes_list
        
        Args: 
        - string/byte value
        
        Returns:
        - tf.train.Feature object as a bytes_list containing the input value.
        """
        if isinstance(value, type(tf.constant(0))): # check if Tf tensor
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def __getitem__(self, batch_index):
        """
        Load the batch from a pre-saved TFRecord file instead of processing raw data.
        Each file contains exactly one batch.
        quantization is done here: Helpful for pretraining without the quantization and the later training with quantized data.
        shuffling is also done here.
        TODO: prefetching (un-done)
        """
        tfrecord_path = self.tfrecord_filenames[batch_index]
        raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        parsed_dataset = raw_dataset.map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

        for X_batch, y_batch in parsed_dataset:
            ''' Add the reshaping in saving'''
            X_batch = tf.reshape(X_batch, [-1, *X_batch.shape[1:]])
            y_batch = tf.reshape(y_batch, [-1, *y_batch.shape[1:]])

            if self.quantize:
                X_batch = QKeras_data_prep_quantizer(X_batch, bits=4, int_bits=0, alpha=1)

            if self.shuffle:
                indices = tf.range(start=0, limit=tf.shape(X_batch)[0], dtype=tf.int32)
                shuffled_indices = tf.random.shuffle(indices, seed=self.seed)
                X_batch = tf.gather(X_batch, shuffled_indices)
                y_batch = tf.gather(y_batch, shuffled_indices)
                
            del raw_dataset, parsed_dataset
            return X_batch, y_batch
            
    @staticmethod
    def _parse_tfrecord_fn(example):
        """
        Parses a single TFRecord example.
        
        Returns:
        - X: as a float32 tensor.
        - y: as a float32 tensor.
        """
        feature_description = {
            'X': tf.io.FixedLenFeature([], tf.string),
            'y': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        X = tf.io.parse_tensor(example['X'], out_type=tf.float32)
        y = tf.io.parse_tensor(example['y'], out_type=tf.float32)
        return X, y

    def __len__(self):
        if len(self.file_offsets) != 1: # used when TFRecord files are created during initialization
            num_batches = self.file_offsets[-1] // self.batch_size
        else: # used during loading saved TFRecord files
            num_batches = len(os.listdir(self.tfrecords_dir))
        return num_batches

    def on_epoch_end(self):
        '''
        This shuffles the file ordering so that it shuffles the ordering in which the TFRecord
        are loaded during the training for each epochs.
        '''
        gc.collect()
        self.epoch_count += 1
        # Log quantization status once
        if self.epoch_count == 1:
            logging.warning(f"Quantization is {self.quantize} in data generator. This may affect model performance.")

        if self.shuffle:
            self.rng.shuffle(self.tfrecord_filenames)
            self.seed += 1 # So that after each epoch the batch is shuffled with a different seed (deterministic)


import tensorflow_probability as tfp

# custom loss function
def custom_loss(p_base, y, minval=1e-9, maxval=1e9, scale = 512):
    
    p = p_base
    
    mu = p[:, 0:8:2]
    
    # creating each matrix element in 4x4
    Mdia = minval + torch.maximum(p[:, 1:8:2], torch.tensor(0.0))
    Mcov = p[:,8:]
    # placeholder zero element
    zeros = torch.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = torch.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = torch.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = torch.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = torch.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = torch.stack([row1,row2,row3,row4])
    scale_tril = scale_tril.permute((2,0,1))
    
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc = mu, scale_tril = scale_tril) 

    likelihood = torch.exp(dist.log_prob(y))
    likelihood = torch.clamp(likelihood,torch.tensor(minval).cuda(), torch.tensor(maxval).cuda())

    NLL = -1*torch.log(likelihood)

    NLL = torch.sum(NLL).cuda()
    return NLL


if __name__=="__main__":
    training_generator = OptimizedDataGenerator(
    data_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/data/",
    labels_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/labels/",
    is_directory_recursive = False,
    file_type = "parquet",
    data_format = "3D",
    batch_size = 5000,
    file_count = 50,
    to_standardize= True,
    include_y_local= False,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    scaling_list = [75.0, 18.75, 10.0, 1.22],
    input_shape = (20,13,21),
    transpose = (0,2,3,1),
    files_from_end=True,
    shuffle= True,
    
    load_from_tfrecords_dir = "train_tfrecords",
    use_time_stamps = -1, #-1
    max_workers = 2, # Don't make this too large (will use up all RAM)
    seed = 10,
    quantize = False # Quantization ON
    )
    validation_generator = OptimizedDataGenerator(
    data_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/data/",
    labels_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/labels/",
    is_directory_recursive = False,
    file_type = "parquet",
    data_format = "3D",
    batch_size = 5000,
    file_count = 10,
    to_standardize= True,
    include_y_local= False,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    scaling_list = [75.0, 18.75, 10.0, 1.22],
    input_shape = (20,13,21),
    transpose = (0,2,3,1),
    files_from_end=True,
    shuffle= True,

    load_from_tfrecords_dir = "val_tfrecords",
    use_time_stamps = -1, #-1
    max_workers = 1, # Don't make this too large (will use up all RAM)
    seed = 10,
    quantize = False # Quantization ON
)
    a, b = next(iter(validation_generator))
    a = tf.transpose(a, (0, 3, 2, 1))
    a = torch.tensor(a.numpy())
    b = torch.tensor(b.numpy())
    print(a.shape)
    print(b.shape)

    model = SmartPixelModel()
    out = model(a)
    from torchinfo import summary
    summary(model, a.shape)
    print(custom_loss(out, b))


def get_smartpixel_data_and_model(data=True):
    """training_generator = OptimizedDataGenerator(
    data_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/data/",
    labels_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/labels/",
    is_directory_recursive = False,
    file_type = "parquet",
    data_format = "3D",
    batch_size = 5000,
    file_count = 50,
    to_standardize= True,
    include_y_local= False,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    scaling_list = [75.0, 18.75, 10.0, 1.22],
    input_shape = (20,13,21),
    transpose = None,
    files_from_end=True,
    shuffle= True,
    
    load_from_tfrecords_dir = "train_tfrecords",
    tfrecords_dir=None,
    use_time_stamps = -1, #-1
    max_workers = 2, # Don't make this too large (will use up all RAM)
    seed = 10,
    quantize = False # Quantization ON
    )
    validation_generator = OptimizedDataGenerator(
    data_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/data/",
    labels_directory_path = "/home/nroope/Documents/smart-pixels-ml/dataset8/unflipped/labels/",
    is_directory_recursive = False,
    file_type = "parquet",
    data_format = "3D",
    batch_size = 5000,
    file_count = 10,
    to_standardize= True,
    include_y_local= False,
    labels_list = ['x-midplane','y-midplane','cotAlpha','cotBeta'],
    scaling_list = [75.0, 18.75, 10.0, 1.22],
    input_shape = (20,13,21),
    transpose = None,
    files_from_end=True,
    shuffle= True,

    load_from_tfrecords_dir = "val_tfrecords",
    use_time_stamps = -1, #-1
    max_workers = 1, # Don't make this too large (will use up all RAM)
    seed = 10,
    quantize = False # Quantization ON
    )"""
    model = SmartPixelModel().to(device="cuda")
    loss_func = custom_loss
    training_dataset_files = glob.glob("data/smartpixel/train/*")

    if not data:
        return model, loss_func
    import random
    random.shuffle(training_dataset_files)
    tdata = []
    vdata = []
    for f in tqdm(training_dataset_files):
        data = np.load(f)
        inputs = data["inputs"].astype(np.float16)
        target = data["target"].astype(np.float16)
        tdata.append((inputs, target))
    
    validation_dataset_files = glob.glob("data/smartpixel/validation/*")
    for f in tqdm(validation_dataset_files):
        data = np.load(f)
        inputs = data["inputs"].astype(np.float16)
        target = data["target"].astype(np.float16)
        vdata.append((inputs, target))
        
        


    return model, tdata, vdata, loss_func

