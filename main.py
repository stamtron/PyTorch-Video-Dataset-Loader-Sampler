from video_dataset import*
from transforms import *

root_dir = './video_data_test/'
class_paths = [d.path for d in os.scandir(root_dir) if d.is_dir]

class_image_paths = []
end_idx = []
for c, class_path in enumerate(class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = sorted(glob.glob(os.path.join(d.path, '*.png')))
            # Add class idx to paths
            paths = [(p, c) for p in paths]
            class_image_paths.extend(paths)
            end_idx.extend([len(paths)])

end_idx = [0, *end_idx]
end_idx = torch.cumsum(torch.tensor(end_idx), 0)
seq_length = 10

sampler = MySampler(end_idx, seq_length)

tensor_transform = get_tensor_transform('ImageNet', False)
train_spat_transform = get_spatial_transform(2)
train_temp_transform = get_temporal_transform()
valid_spat_transform = get_spatial_transform(0)
valid_temp_transform = va.TemporalFit(size=16)

dataset = MyDataset(
        image_paths = class_image_paths,
        seq_length = seq_length,
        temp_transform = train_temp_transform,
        spat_transform = train_spat_transform,
        tensor_transform = tensor_transform,
        length = len(sampler),
        lstm = False,
        oned = False,
        augment = False,
        multi = 1)
        
loader = DataLoader(
        dataset,
        batch_size = 1,
        sampler = sampler,
        drop_last = True,
        num_workers = 0)
        
for data, target in loader:
    print(data.shape)
