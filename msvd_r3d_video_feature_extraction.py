from pathlib import Path
from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from tqdm import tqdm

from prepare_msvd_dataset import MSVDDataset
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class R3DFeatureExtractionDataset(Dataset):

    def __init__(self, ids, path, im_size=112):
        self.ids = ids
        self.path = path
        self.transform = Compose([Resize(im_size), CenterCrop(im_size),
                                  Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]), ])
        self.im_size = im_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        path = self.ids[index]
        visual, audial, info = read_video(str(path), pts_unit="sec", output_format="TCHW")

        visual_frames = torch.zeros((visual.shape[0], visual.shape[1], self.im_size, self.im_size))
        for i in range(visual.shape[0]):
            im = visual[i]
            im = torch.div(im, 255)
            im = self.transform(im)
            visual_frames[i] = im

        id_parts = str(path.parts[-1][:-4])
        return visual_frames, id_parts


class R3DFeatureExtraction:
    def __init__(self, input_folder: Path, output_folders: List[Union[Path, Path]], model: torch.nn.Module,
                 image_size: int, parameters: dict, maximum_len_of_visual) -> None:

        self.input_folder = input_folder  # input video folder
        self.output_visual_folder = output_folders  # output folder for visual features
        self.model = model.eval()  # assign model in evaluation mode
        self.IMAGE_SIZE = image_size  # initialize input image size
        self.params = parameters  # initialize parameters for dataloader
        self.ids = list(input_folder.glob('*.*'))  # list of video files from input folder
        self.set = R3DFeatureExtractionDataset(self.ids, self.input_folder, self.IMAGE_SIZE)  # initialize dataset
        self.generator = DataLoader(self.set, **self.params)  # initialize dataloader
        self.max_visual_len = maximum_len_of_visual
        self.patch_size = 32

        if not r3d_output_folder.exists():
            r3d_output_folder.mkdir(parents=True, exist_ok=True)
            print(f"path: {str(r3d_output_folder)} is created.")

    def run(self) -> None:
        print(f"Extracting: {str(self.input_folder)}")
        max_frame_length = 0
        max_seq_length = 0

        for local_visual, local_ids in tqdm(self.generator):
            local_visual = local_visual[0].to(DEVICE)  # T x C x H x W
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
                    x = self.model(x.unsqueeze(0))
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

        # Sanity check to ensure that all feature files have the same sequence length
        for feature_file in self.output_visual_folder.glob("*.pt"):
            visual_feature = torch.load(feature_file)
            sequence_length = visual_feature.shape[-1]
            assert sequence_length == max_seq_length, f"File {feature_file} has an invalid sequence length"


if __name__ == '__main__':
    dt = MSVDDataset()
    r3d_output_folder = dt.train_visual_r3d_feature_folder
    input_folder = dt.train_folder
    r3d_model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    r3d_model = r3d_model.to(DEVICE)
    r3d_model = r3d_model.eval()

    params = {'batch_size': 1, 'shuffle': False, 'num_workers': 0}
    IMAGE_SIZE = 112
    max_visual_len = 32
    r3d = R3DFeatureExtraction(dt.train_folder, r3d_output_folder, r3d_model, IMAGE_SIZE, params, max_visual_len)
    r3d.run()
