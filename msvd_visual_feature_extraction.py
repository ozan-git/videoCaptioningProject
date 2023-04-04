# Tries for dataloader and dataset on feature extraction.

from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.models.video import s3d, S3D_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm

from prepare_msvd_dataset import MSVDDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

pretrained_model = s3d(weights=S3D_Weights.KINETICS400_V1)
pretrained_model = pretrained_model.to(DEVICE)
pretrained_model = pretrained_model.eval()

# Parameters
PARAMS = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 0}


class FeatureExtractionDataset(Dataset):
    """Characterizes a dataset for PyTorch"""
    '''
        A Torch Dataset utilized to Load Video's visual and audial data.
        inputs are:
        ids: list of paths where the video files are.
        if the ids are consist of only names.
        path: to determine where are the files.
        im_size: To transform the image into specified size. ie: inception_v3 model takes 3x299x299 so the im_size = 299.
        frame_size: how many visual frame to be extracted from the visual data.
    '''

    def __init__(self, ids, path, im_size=224):
        """Initialization"""
        self.ids = ids
        self.path = path
        self.transform = Compose([
            Resize(im_size),
            CenterCrop(im_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.im_size = im_size

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.ids)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample

        path = self.ids[index]
        # Load data as visual, audial, and get fps of both visual and audial.
        visual, audial, info = read_video(str(path), pts_unit="sec", output_format="TCHW")

        # initialize visual frames with zeros.
        visual_frames = torch.zeros((visual.shape[0], visual.shape[1], self.im_size, self.im_size))
        # for loop to get every determined frame with determined step size.
        for i in range(visual.shape[0]):
            # get the frame from the visual data.
            im = visual[i]
            # normalize to 0-1.
            im = torch.div(im, 255)
            # Transform the images into given im size and normalize.
            im = self.transform(im)
            # add the resulted frame into the corresponding visual frames index.
            visual_frames[i] = im

        id = str(path.parts[-1][:-4])  # get ID of the image from its path.

        # visual_frames = visual_frames.permute(1, 0, 2, 3)

        return visual_frames, id


class FeatureExtraction:
    """
    Class for visual and audial feature extraction.
    """

    def __init__(self, input_folder: Path, output_folders: List[Union[Path, Path]], model: torch.nn.Module,
                 IMAGE_SIZE: int, params: dict, max_visual_len) -> None:

        self.input_folder = input_folder  # input video folder
        self.output_visual_folder = output_folders  # output folder for visual features
        self.model = model.eval()  # assign model in evaluation mode
        self.IMAGE_SIZE = IMAGE_SIZE  # initialize input image size
        self.params = params  # initialize parameters for dataloader
        self.ids = list(input_folder.glob('*.*'))  # list of video files from input folder
        self.set = FeatureExtractionDataset(self.ids, self.input_folder, self.IMAGE_SIZE)  # initialize dataset
        self.generator = DataLoader(self.set, **self.params)  # initialize dataloader
        self.max_visual_len = max_visual_len
        self.patch_size = 250
        # create output folders
        if not self.output_visual_folder.exists():
            Path(self.output_visual_folder).mkdir(parents=True, exist_ok=True)
            print(f"path: {str(self.output_visual_folder)} is created.")

    def run(self) -> None:
        """
            Scan through generator, extract features from visual data
            from videos and save them in .pt files in corresponding output folders.
        """
        print(f"Extracting: {str(self.input_folder)}")
        max_frame_length = 0
        max_seq_length = 0

        for local_visual, local_ids in tqdm(self.generator):
            local_visual = local_visual[0].to(DEVICE)  # T x C x H x W
            # T x C x H x W mean T is the number of frames, C is the number of channels,
            # H is the height, and W is the width.
            frame_length = local_visual.shape[0]

            if frame_length > max_frame_length:
                max_frame_length = frame_length

            feature_path = self.output_visual_folder / f"{local_ids[0]}.pt"
            len_of_patches = (frame_length // self.patch_size) + 1
            visual_feature = None
            sequence_length = 0

            for i in range(len_of_patches):
                start_idx = i * self.patch_size
                end_idx = start_idx + self.patch_size
                if end_idx > local_visual.shape[0]:
                    x = torch.zeros(
                        (self.patch_size, local_visual.shape[1], local_visual.shape[2], local_visual.shape[3]),
                        device=DEVICE)
                    x[:local_visual.shape[0] - start_idx, :, :, :] = local_visual[start_idx:]
                else:
                    x = local_visual[start_idx:end_idx]
                x = x.permute(1, 0, 2, 3)
                with torch.no_grad():
                    x = self.model.features(x.unsqueeze(0))
                    x = self.model.avgpool(x)
                    x = x.squeeze()
                    sequence_length += x.shape[-1]
                    if visual_feature is None:
                        visual_feature = x
                    else:
                        visual_feature = torch.cat((visual_feature, x), dim=-1)
            if sequence_length > max_seq_length:
                max_seq_length = sequence_length
            torch.save(visual_feature.squeeze().cpu(), feature_path)
        print(f"{str(self.input_folder)} extracted.")
        print(f"Maximum frame length: {max_frame_length}")
        print(f"Maximum sequence length: {max_seq_length}")


dt = MSVDDataset()

extract = FeatureExtraction(dt.train_folder, dt.train_visual_features_folder, model=pretrained_model, IMAGE_SIZE=224,
                            params=PARAMS, max_visual_len=0)
extract.run()
