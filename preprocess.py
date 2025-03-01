import os
import nibabel as nib
import numpy as np
import pickle
import random 
from pyparsing import line
import torch
from torchvision.transforms import transforms
from torch import nn
from typing import List, Any, Dict
import SimpleITK as sitk

def nib_load(file_name):
    """
     Load nifti file and convert to numpy array
    """
    if not os.path.exists(file_name):
        raise ValueError(f'File not found: {file_name}')
    
    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

def n4itk(image_np):
    """
    Apply N4ITK bias field correction to a 3D numpy array
    """
    # Convert numpy array to SimpleITK image
    image_sikt = sitk.GetImageFromArray(image_np.astype(np.float32))

    # Create mask - N4 works better with a mask of the non-zero regions
    mask_np = (image_np > 0).astype(np.uint8)
    mask_sitk = sitk.GetImageFromArray(mask_np)

    # Initialize N4 bias field correction filter
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    # Number of iterations at each resolution level
    corrector.SetMaximumNumberOfIterations([75, 50, 25])

    # Convergence threshold (stopping criterion)
    corrector.SetConvergenceThreshold(0.0005)

    # Apply correction
    try:
        corrected_image_sitk = corrector.Execute(image_sikt, mask_sitk)
        corrected_image_np = sitk.GetArrayFromImage(corrected_image_sitk)
        return corrected_image_np
    except Exception as e:
        print(f"Warning: N4 bias field correction failed: {e}")
        print("Returning original image")
        return image_np


def load_data(ROOT_PATH, save_path):
    """
    Save each case to one pickle file
    """
    modalities = ('flair', 't1ce', 't1', 't2')
    case_names = os.listdir(ROOT_PATH)
    total_loaded = 0
    
    # Create save path for pickle files
    os.makedirs(save_path, exist_ok=True)
    
    # Save case one by one
    for case in case_names:
        try:
            # Load the data of single case
            img = np.stack([
                nib_load(os.path.join(ROOT_PATH, case, f"{case}_{mod}.nii.gz")).astype('float32') 
                for mod in modalities
            ], axis=-1)

            # Now apply N4ITK correction to each modality in the stacked array
            for k in range(img.shape[-1]):  # For each modality
                print(f"Applying N4ITK bias field correction to {case}_{modalities[k]}.nii.gz")
                img[..., k] = n4itk(img[..., k])

            # Normalize the non-zero area
            mask = img.sum(-1) > 0
            for k in range(4):
                x = img[..., k]
                y = x[mask]
                
                x[mask] -= y.mean()
                x[mask] /= y.std()
                
                img[..., k] = x
            
            label = nib_load(os.path.join(ROOT_PATH, case, f"{case}_seg.nii.gz")).astype('uint8')
            
            # Create pickle file for one case
            case_save_path = os.path.join(save_path, f"{case}.pkl")
            pksave((img, label), case_save_path)
            
            print(f'Successfully loaded and saved {case}')
            total_loaded += 1
            
            # Clean memory
            del img, label
            
        except Exception as e:
            print(f"Error processing {case}: {str(e)}")
            continue
    
    # Create index file
    index_path = os.path.join(save_path, 'processed_cases.txt')
    with open(index_path, 'w') as f:
        f.write('\n'.join(sorted([case for case in os.listdir(save_path) if case.endswith('.pkl')])))
    
    print(f"{len(case_names)} are expected to load")
    print(f"{total_loaded} are loaded")

class Random_crop_3d(object):
    """Random crop augmentation"""
    def __call__(self, data):
        image = data['image']
        label = data['label']
        
        H = random.randint(0, 240 - 128)
        W = random.randint(0, 240 - 128)
        D = random.randint(0, 160 - 128)
        
        image = image[H: H + 128, W: W + 128, D: D + 128, ...]
        label = label[..., H: H + 128, W: W + 128, D: D + 128]
        
        data['image'] = image
        data['label'] = label
        return data

class Random_flip(object):
    """Random flip augmentation"""
    def __call__(self, data):
        
        image = data['image']
        label = data['label']
        
        if random.random() < 0.5:
            image = np.flip(image, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            image = np.flip(image, 2)
            label = np.flip(label, 2)
            
        data['image'] = image
        data['label'] = label
        return data

class Random_intensity_shift(object):
    """Random intensity shift augmentation"""
    def __call__(self, data, factor=0.1):
        image = data['image']
        
        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, 
                                    size=[1, image.shape[1], 1, image.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, 
                                    size=[1, image.shape[1], 1, image.shape[-1]])
        
        image = image * scale_factor + shift_factor
        data['image'] = image
        return data
    
class Pad(object):
    def __call__(self, data):
        image = data['image']
        label = data['label']
        
        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return {'image': image, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data):
        image = data['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        label = data['label']
        label = np.ascontiguousarray(label)
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        return {'image': image, 'label': label}
    
def split_data(ROOT_PATH, pk_path, train_ratio = 0.8, seed = 42):
    """
    Split the data into training and testing set.
    The data is downloaded from kaggle which only provided the training set.
    In order to avoid performance degradation due to reduced training data size, there is no validation set

    Returns:
    train.txt and test.txt: The file name of the training and testing datasets' files
    """
    random.seed(seed)
    np.random.seed(seed)
    modalities = ('flair', 't1ce', 't1', 't2') 
    
    # Check if all modalities is correctly loaded
    case_namne = []
    for subdir in os.listdir(ROOT_PATH):
        case_path = os.path.join(ROOT_PATH, subdir)
        if os.path.isdir(case_path):
            if all(os.path.exists(os.path.join(case_path, f'{subdir}_{modal}.nii.gz')) for modal in modalities):
                case_namne.append(subdir)

    random.shuffle(case_namne)

    # Split into training set and testing set
    train_size = int(len(case_namne)* train_ratio)
    train_cases = case_namne[: train_size]
    test_cases = case_namne[train_size: ]

    #Create directory
    train_path = os.path.join(pk_path, 'TRAIN')
    test_path = os.path.join(pk_path, 'TEST')
    os.makedirs(train_path, exist_ok = True)
    os.makedirs(test_path, exist_ok = True)
    
    # Write into file
    with open(os.path.join(train_path, 'train.text'), 'w') as f:
        for case in train_cases:
            f.write(f'{case}\n')

    with open(os.path.join(test_path, 'test.text'), 'w') as f:
        for case in test_cases:
            f.write(f'{case}\n')
    
    print(f'Data splited: {len(train_cases)} training samples, {len(test_cases)} testing samples')

def load_and_convert_to_dict(path):
    """
    Read tuple from pickle file and return it as dictionary
    """
    image, label = pkload(path)
    dic = {'image': image, 'label': label}
    return dic
    
def preprocess_train(path):
    """
    Preprocess the training set
    """
    dic = load_and_convert_to_dict(path)

    pre = transforms.Compose([
        Pad(),
        Random_crop_3d(),
        Random_flip(),
        Random_intensity_shift(),
        ToTensor()
    ])
    
    return pre(dic)

def preprocess_test(path):
    """
    Preprocess the test data
    """
    dic = load_and_convert_to_dict(path)

    pre = transforms.Compose([
        Pad(),
        ToTensor()
    ])

    return pre(dic)

def pkload(file_path):
    """ Load a single pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
        
def pksave(data, file_path):
    """ Save the data into a pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

class PicklePreProcessor:
    """
    Class for batch processing of pickle files, 
    supporting different training and test set processing methods

    Parameters:
            train_paths_file: Text file containing names to training pickle files
            test_paths_file: Text file containing names to test pickle files
            processed_train_dir: Directory to save processed training files
            processed_test_dir: Directory to save processed test files
            base_diur: Directoy to save the path of un-preprocessed pickle files
    """
    def __init__(self, train_paths_file: str = None, test_paths_file: str = None,
                 base_dir: str = None):
        self.train_paths_file = train_paths_file
        self.test_paths_file = test_paths_file
        self.processed_train_dir = os.path.dirname(train_paths_file)
        self.processed_test_dir = os.path.dirname(test_paths_file)
        self.train_paths = []
        self.test_paths = []
        self.base_dir = base_dir

    def load_paths(self,is_train: bool = True) -> List[str]:
        """
        Load pickle file path from txt file

        Args: 
        is_train: whether to load the training set file path

        Returns:
        Pickles file path list
        """

        paths_file = self.train_paths_file if is_train else self.test_paths_file
        paths_list = self.train_paths if is_train else self.test_paths

        if not paths_file:
            print(f"{'Training' if is_train else 'Testing'} path file not specified")
            return []
        
        try:
            with open(paths_file, 'r', encoding= 'utf-8') as f:
                paths_list = [os.path.join(self.base_dir, line.strip() + '.pkl') for line in f if line.strip()]
            if is_train:
                self.train_paths = paths_list
            else:
                self.test_paths = paths_list
            return paths_list
        
        except Exception as e:
            print(f"Error loading {'training' if is_train else 'test'} path file: {e}")
            return []
        
    def process_files(self, is_train: bool = True) -> List[Any]:
        """
        Process all pickle files of specified type
        
        Parameters:
            is_train: Whether to process training data
            
        Returns:
            List of processing results
        """
        paths_list = self.train_paths if is_train else self.test_paths
        dataset_type = "Training" if is_train else "Test"
        output_dir = self.processed_train_dir if is_train else self.processed_test_dir

        success_count = 0
        error_count = 0
        
        # If path list is empty, try to load
        if not paths_list:
            paths_list = self.load_paths(is_train)
            if not paths_list:
                print(f"{dataset_type} path is empty, cannot process")
                return []
        
        processor_func = preprocess_train if is_train else preprocess_test
        total_files = len(paths_list)
            
        # Process each file
        for i, path in enumerate(paths_list):
            try:
                # Check if file exists
                if not os.path.exists(path):
                    print(f"File does not exist: {path}")
                    continue
                
                # Process data
                result = processor_func(path)

                if os.path.exists(output_dir):
                    filename = os.path.basename(path)
                    output_path = os.path.join(output_dir, filename)
                    pksave(result, output_path)
                    print(f"Saved processed file to: {output_path}")
                    success_count += 1
                else:
                    print(f'{output_dir} does not exist')
                
                print(f"Processed {dataset_type} [{i+1}/{len(paths_list)}]: {path}")
                                
            except Exception as e:
                print(f"Error processing {dataset_type} file {path}: {e}")
                error_count += 1
        
        print(f"\n--- {dataset_type} Processing Summary ---")
        print(f"Total files: {total_files}")
        print(f"Successfully processed: {success_count} ({success_count/total_files*100:.1f}%)")
        print(f"Errors encountered: {error_count} ({error_count/total_files*100:.1f}%)")
                
    
    def process_all(self) -> Dict[str, List[Any]]:
        """
        Process all training and test files
        
        Returns:
            Dictionary containing results for training and test data
        """
        self.process_files(is_train=True)
        self.process_files(is_train=False)

        print("Processing completed")
        if self.processed_train_dir:
            print(f"Training files saved to: {self.processed_train_dir}")
        if self.processed_test_dir:
            print(f"Test files saved to: {self.processed_test_dir}")