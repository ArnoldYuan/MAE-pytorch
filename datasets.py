import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import PIL
import os.path as osp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from masking_generator import RandomMaskingGenerator
from dataset_folder import ImageFolder


class CelebA(Dataset):
    def __init__(self, split, img_path='~/CelebA/celeba/img_align_celeba/', identity_file='~/CelebA/celeba/identity_CelebA.txt', num_ids=1000, trans=False):
        self.num_ids = num_ids
        self.trans = trans
        self.img_path = osp.expanduser(img_path)
        with open(osp.expanduser(identity_file)) as f:
            lines = f.readlines()

        id2file = {}
        for line in lines:
            file, id = line.strip().split()
            id = int(id)
            if id in id2file.keys():
                id2file[id].append(file)
            else:
                id2file[id] = [file]

        thres = 25
        id2file_cleaned = {}
        for key in id2file.keys():
            if len(id2file[key]) > thres:
                id2file_cleaned[key] = id2file[key]

        self.name_list = []
        self.label_list = []

        if split == 'pub':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:2000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:2000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub1':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:1000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub1-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[:1000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub2':
            i = 0
            for key in sorted(id2file_cleaned.keys())[1000:2000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pub2-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[1000:2000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pri':
            i = 0
            for key in sorted(id2file_cleaned.keys())[2000:3000]:
                for file in id2file_cleaned[key][:20]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        elif split == 'pri-dev':
            i = 0
            for key in sorted(id2file_cleaned.keys())[2000:3000]:
                for file in id2file_cleaned[key][20:25]:
                    self.name_list.append(file)
                    self.label_list.append(i)
                i += 1
        else:
            raise NotImplementedError()

        self.processor = self.get_processor()
    
    def get_processor(self):
        crop_size = 108
        re_size = 224
        offset_height = (218 - crop_size) // 2
        offset_width = (178 - crop_size) // 2
        crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

        proc = []
        proc.append(transforms.ToTensor())
        proc.append(transforms.Lambda(crop))
        proc.append(transforms.ToPILImage())
        proc.append(transforms.Resize((re_size, re_size)))
        proc.append(transforms.ToTensor())
                    
        return transforms.Compose(proc)

    def __getitem__(self, index):
        path = self.img_path + "/" + self.name_list[index]
        img = PIL.Image.open(path).convert('RGB')
        img = self.processor(img)
        label = self.label_list[index]

        return img, label

    def __len__(self):
        return len(self.name_list)


class DataAugmentationForMAE(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, args.mask_ratio
        )

    def __call__(self, image):
        return self.transform(image), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForMAE(args)
    print("Data Aug = %s" % str(transform))
    return ImageFolder(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'CelebA':
        dataset = CelebA(split='pri') if is_train else CelebA(split='pri-dev')
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
