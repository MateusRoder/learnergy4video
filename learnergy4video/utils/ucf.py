import glob
import os
import numpy as np
import torch
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class UCF101(VisionDataset):
    """
    `UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_ dataset.

    UCF101 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the UCF101 Dataset.
        annotation_path (str): path to the folder containing the split files
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1,
                 frame_rate=None, fold=1, train=True, transform=None,
                 _precomputed_metadata=None, num_workers=1, _video_width=0,
                 _video_height=0, _video_min_dimension=0, _audio_samples=0):
        super(UCF101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        extensions = ('avi',)
        self.fold = fold
        self.train = train
        dy = 240
        dx = 320
        self.d_y = int(.3*dy)
        self.d_x = int(.3*dx) 
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )
        self.video_clips_metadata = video_clips.metadata
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips_metadata

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        if train:
            name = "{}list{:02d}.txt".format(name, fold)
            f = os.path.join(annotation_path, name)
            selected_files = []
            labels = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                lb = [x[1] for x in data]
                data = [x[0] for x in data]
                selected_files.extend(data)
                labels.extend(lb)

            self.label = labels
            selected_files = set(selected_files)
            indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:] in selected_files]
        else:
            name_class = "classInd.txt"
            f = os.path.join(annotation_path, name_class)
            lb = []
            n_file = []
            with open(f, "r") as fid:
                data = fid.readlines()
                data = [x.strip().split(" ") for x in data]
                lb2 = [x[0] for x in data]
                data = [x[1] for x in data]
                lb.extend(lb2)
                n_file.extend(data)

            name = "{}list{:02d}.txt".format(name, fold)
            f = os.path.join(annotation_path, name)
            selected_files = []
            labels = []
            with open(f, "r") as fid:
                data = fid.readlines()
                nome = [x.split("/") for x in data]
                data = [x.strip().split(" ") for x in data]
                nome = [x[0] for x in nome]
                data = [x[0] for x in data]
                for i in range(len(n_file)):
                    for j in range(len(nome)):
                        if nome[j]==n_file[i]:
                            labels.append(lb[i])
                selected_files.extend(data)
            self.label = labels
            selected_files = set(selected_files)
            indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:] in selected_files]
        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        vd = torch.zeros((video.size(0), self.d_y, self.d_x))
        #label = self.samples[self.indices[video_idx]][1]
        label = int(self.label[video_idx])
        
        for i in range(video.size(0)):
            a_ = video[i].numpy()
            im = Image.fromarray(a_).convert('RGB')#.resize((self.d_x, self.d_y))    
            im = Image.fromarray(np.array(im)).convert('LA').resize((self.d_x, self.d_y))
            a_ = np.array(im)[:,:,0]
            vd[i,:,:] = torch.from_numpy(a_)

        if self.transform is not None:
            video = self.transform(video)

        return vd, audio, label
