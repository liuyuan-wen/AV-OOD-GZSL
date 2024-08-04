import logging
import pickle
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import sys
from src.model_addition import map_label
from src.utils import read_features, get_class_names


class Dataset(data.Dataset):
    def __init__(self, args, dataset_split, syn_data=None, verbose=True):
        super(Dataset, self).__init__()
        self.args = args
        self.logger = logging.getLogger()
        self.root = args.dataset.root_set / args.dataset.dataset_name
        self.dataset_name = args.dataset.dataset_name
        self.feature_extraction_method = args.dataset.feature_extraction_method
        self.zero_shot_split = args.dataset.zero_shot_split
        self.dataset_split = dataset_split
        self.verbose = verbose
        self.syn_data = syn_data

        self.logger.info(
            f"Initializing Dataset {args.dataset.dataset_name}\t"
            f"Dataset split: {dataset_split}")
            
        self.preprocess()
        self.data = self.get_data()

        if self.syn_data is not None:
            if self.dataset_split == "train_val_syn":
                self.data['audio']['data'] = torch.cat((self.data['audio']['data'], syn_data['audio']['data']), dim=0)
                self.data['audio']['target'] = torch.cat((self.data['audio']['target'], syn_data['audio']['target']), dim=0)
                self.data['video']['data'] = torch.cat((self.data['video']['data'], syn_data['video']['data']), dim=0)
                self.data['video']['target'] = torch.cat((self.data['video']['target'], syn_data['video']['target']), dim=0)
                self.data["audio"]["url"] = np.concatenate((self.data["audio"]["url"], np.ones(syn_data['audio']['data'].shape[0])))
            elif self.dataset_split == "syn":
                self.data['audio']['data'] = syn_data['audio']['data']
                self.data['audio']['target'] = syn_data['audio']['target']
                self.data['video']['data'] = syn_data['video']['data']
                self.data['video']['target'] = syn_data['video']['target']
                self.data["audio"]["url"] = np.ones(syn_data['audio']['data'].shape[0])
            else:
                raise AttributeError("Dataset split not correct.")

    @property
    def all_targets(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.all_class_ids))[0]
        return self.data["audio"]["target"][classes_mask]

    @property
    def all_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.seen_unseen_class_ids))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "cat": torch.cat((self.data["audio"]["data"][classes_mask], self.data["video"]["data"][classes_mask]), dim=1),
            "text": self.data["text"]["data"][self.seen_unseen_class_ids],
            "target": self.data["audio"]["target"][classes_mask],
            "target_mapped": map_label(self.data["audio"]["target"][classes_mask], self.seen_unseen_class_ids)[0],
            "ids": self.seen_unseen_class_ids,
            "url": self.data["audio"]["url"][classes_mask]
        }
    
    @property
    def seen_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.seen_class_ids))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "cat": torch.cat((self.data["audio"]["data"][classes_mask], self.data["video"]["data"][classes_mask]), dim=1),
            "text": self.data["text"]["data"][self.seen_class_ids],
            "target": self.data["audio"]["target"][classes_mask],
            "target_mapped": map_label(self.data["audio"]["target"][classes_mask], self.seen_class_ids)[0],
            "ids": self.seen_class_ids,
            "url": self.data["audio"]["url"][classes_mask]
        }
    
    @property
    def unseen_data(self):
        classes_mask = np.where(np.isin(self.data["audio"]["target"], self.unseen_class_ids))[0]
        return {
            "audio": self.data["audio"]["data"][classes_mask],
            "video": self.data["video"]["data"][classes_mask],
            "cat": torch.cat((self.data["audio"]["data"][classes_mask], self.data["video"]["data"][classes_mask]), dim=1),
            "text": self.data["text"]["data"][self.unseen_class_ids],
            "target": self.data["audio"]["target"][classes_mask],
            "target_mapped": map_label(self.data["audio"]["target"][classes_mask], self.unseen_class_ids)[0],
            "ids": self.unseen_class_ids,
            "url": self.data["audio"]["url"][classes_mask]
        }

    @property
    def map_embeddings_target(self):
        w2v_embedding = self.data["text"]["data"][self.seen_unseen_class_ids].cuda()
        sorted_classes = self.seen_unseen_class_ids
        mapping_dict = {}
        for i in range(len(sorted_classes)):
            mapping_dict[int(sorted_classes[i])] = i
        return w2v_embedding, mapping_dict

    @property
    def features_processed_folder(self):
        return Path().cwd() / self.root / "_features_processed"

    @property
    def all_class_ids(self):
        return np.sort(np.asarray([self.class_to_idx[name.lower()] for name in self.all_class_names]))

    @property
    def seen_class_ids(self):
        return np.sort(np.asarray([self.class_to_idx[name.lower()] for name in self.seen_class_names]))

    @property
    def unseen_class_ids(self):
        return np.sort(np.asarray([self.class_to_idx[name.lower()] for name in self.unseen_class_names]))

    @property
    def seen_unseen_class_ids(self):
        return np.sort(np.concatenate((self.seen_class_ids, self.unseen_class_ids)))

    @property
    def text_label_mapping(self):
        df = pd.read_csv(self.root / f"class-split/{self.zero_shot_split}/{self.dataset_name.lower()}_w2v_class_names.csv")
        return {val: df.original[idx] for idx, val in enumerate(df.manual)}

    @property
    def class_to_idx(self):
        return {_class.lower(): i for i, _class in enumerate(sorted(self.all_class_names))}

    @property
    def seen_class_names(self):
        if self.dataset_split == "train":
            return self.train_class_names
        elif self.dataset_split == "val":
            return self.val_seen_class_names
        elif self.dataset_split == "train_val":
            return np.concatenate((self.train_class_names, self.val_unseen_class_names))
        elif self.dataset_split == "test":
            return self.test_seen_class_names
        elif self.dataset_split == "train_val_syn":
            if self.syn_data is not None and self.syn_data['audio']['data'].shape[0] > 0:
                return np.concatenate((self.train_class_names, self.val_unseen_class_names, self.test_unseen_class_names))
            elif self.syn_data is not None and self.syn_data['audio']['data'].shape[0] == 0:
                return np.concatenate((self.train_class_names, self.val_unseen_class_names))
            else:
                raise AttributeError("No synthetic data provided.")
        elif self.dataset_split == "syn":
            if self.syn_data is not None:
                return self.test_unseen_class_names
            else:
                raise AttributeError("No synthetic data provided.")
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test,train_val_syn,syn}")

    @property
    def all_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/all_class.txt")

    @property
    def unseen_class_names(self):
        if self.dataset_split == "train":
            return np.array([])
        elif self.dataset_split == "val":
            return self.val_unseen_class_names
        elif self.dataset_split == "train_val":
            return np.array([])
        elif self.dataset_split == "test":
            return self.test_unseen_class_names
        elif self.dataset_split == "train_val_syn":
            return np.array([])
        elif self.dataset_split == "syn":
            return np.array([])
        else:
            raise AttributeError("Dataset split has to be in {train,val,train_val,test}")

    @property
    def train_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_train.txt")

    @property
    def val_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_seen.txt")

    @property
    def val_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_1_val_unseen.txt")

    @property
    def train_val_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_train.txt")

    @property
    def test_seen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_seen.txt")

    @property
    def test_unseen_class_names(self):
        return get_class_names(self.root / f"class-split/{self.zero_shot_split}/stage_2_test_unseen.txt")

    @property
    def training_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"training{self.zero_shot_split}.pkl"

    @property
    def val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"val{self.zero_shot_split}.pkl"

    @property
    def train_val_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"train_val{self.zero_shot_split}.pkl"

    @property
    def test_file(self):
        return self.features_processed_folder / self.feature_extraction_method / f"test{self.zero_shot_split}.pkl"

    def _check_exists(self):
        return self.training_file.exists() and self.val_file.exists() and self.test_file.exists() and self.train_val_file.exists()

    def preprocess(self):
        if self._check_exists():
            return

        (self.features_processed_folder / self.feature_extraction_method).mkdir(parents=True, exist_ok=True)

        self.logger.info('Processing extracted features for faster training (only done once)...')
        self.logger.info(
            f"Processed files will be stored locally in {(self.features_processed_folder / self.feature_extraction_method).resolve()}"
        )

        training_set = self.read_dataset(dataset_type="train")
        val_set = self.read_dataset(dataset_type="val")
        train_val_set = self.read_dataset(dataset_type="train_val")
        test_set = self.read_dataset(dataset_type="test")

        with self.training_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.training_file}")
            pickle.dump(training_set, f, pickle.HIGHEST_PROTOCOL)
        with self.val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.val_file}")
            pickle.dump(val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.train_val_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.train_val_file}")
            pickle.dump(train_val_set, f, pickle.HIGHEST_PROTOCOL)
        with self.test_file.open('wb') as f:
            self.logger.info(f"Dumping to {self.test_file}")
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)

        if not self._check_exists():
            raise RuntimeError("Dataset not found after preprocessing!")
        self.logger.info("Successfully finished preprocessing.")

    def get_data(self):
        if self.dataset_split == "train":
            data_file = self.training_file
        elif self.dataset_split == "val":
            data_file = self.val_file
        elif self.dataset_split == "train_val":
            data_file = self.train_val_file
        elif self.dataset_split == "test":
            data_file = self.test_file
        elif self.dataset_split == "train_val_syn" or self.dataset_split == "syn":
            data_file = self.train_val_file
        else:
            raise AttributeError("Dataset_split has to be either train, val or test.")

        load_path = (self.features_processed_folder / data_file).resolve()
        if self.verbose:
            self.logger.info(f"Loading processed data from disk from {load_path}")
        with load_path.open('rb') as f:
            return pickle.load(f)

    def read_dataset(self, dataset_type):
        result_audio = self.get_data_by_modality(modality="audio", dataset_type=dataset_type)
        result_video = self.get_data_by_modality(modality="video", dataset_type=dataset_type)
        assert torch.equal(result_audio["target"], result_video["target"])
        assert np.array_equal(result_audio["url"], result_video["url"])
        result_text = self.get_data_by_modality(modality="text", dataset_type=dataset_type)
        return {"audio": result_audio, "video": result_video, "text": result_text}

    def get_data_by_modality(self, modality, dataset_type="train"):
        result = {"data": [], "target": [], "url": []}
        if modality == "text":
            data_raw = np.load(
                (
                        self.root / "features" / self.feature_extraction_method / f"text/word_embeddings_{self.args.dataset.dataset_name.lower()}_normed.npy").resolve(),
                allow_pickle=True).item()
            data_raw_sorted = dict(sorted(data_raw.items()))
            result["data"] = list(data_raw_sorted.values())
            result["target"] = [self.class_to_idx[self.text_label_mapping[key].lower()] for key in
                                list(data_raw_sorted.keys())]

        elif modality == "audio" or modality == "video":
            split_names = []
            if dataset_type == "train":
                split_names.append("stage_1_train")
            elif dataset_type == "val":
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "train_val":
                split_names.append("stage_1_train")
                split_names.append("stage_1_val_seen")
                split_names.append("stage_1_val_unseen")
            elif dataset_type == "test":
                split_names.append("stage_2_test_seen")
                split_names.append("stage_2_test_unseen")
            else:
                raise AttributeError("Dataset type incompatible. Has to be either train, val or test.")

            for split_name in split_names:
                modality_path = (
                        self.root / "features" / self.feature_extraction_method / f"{modality}/{split_name}").resolve()
                files = modality_path.iterdir()
                for file in tqdm(files, total=len(list(modality_path.glob('*'))),
                                 desc=f"{dataset_type}:{modality}:{split_name}"):
                    data, url = read_features(file)
                    for i, d in enumerate(data):
                        result["data"].append(d)
                        result["target"].append(self.class_to_idx[file.stem.lower()])
                        result["url"].append(url[i])
        else:
            raise AttributeError("Modality has to be either audio, video or text")
        result["data"] = torch.FloatTensor(result["data"])
        result["target"] = torch.LongTensor(result["target"])
        result["url"] = np.array(result["url"])
        return result



class ContrastiveDataset(data.Dataset):
    def __init__(self, zsl_dataset, verbose=True):
        super(ContrastiveDataset, self).__init__()
        self.logger = logging.getLogger()
        self.verbose = verbose
        if self.verbose:
            self.logger.info(
                f"Initializing Dataset {self.__class__.__name__}\t"
                f"Based on Dataset: {zsl_dataset.__class__.__name__}\t"
                f"with split: {zsl_dataset.dataset_split}")
        self.zsl_dataset = zsl_dataset
        self.dataset_split = self.zsl_dataset.dataset_split
        self.seen_unseen_class_ids = self.zsl_dataset.seen_unseen_class_ids
        self.seen_class_ids = self.zsl_dataset.seen_class_ids
        self.unseen_class_ids = []

        self.data = self.zsl_dataset.all_data
        self.targets = self.zsl_dataset.all_data["target"]
        self.targets_set = set(self.targets.tolist())
        self.target_to_indices = {target: np.where(self.targets == target)[0]
                                    for target in self.targets_set}

        random_state = np.random.RandomState(29)
        pos_neg_pairs = [[i,
                            random_state.choice(self.target_to_indices[
                                                    np.random.choice(
                                                        list(self.targets_set - set([self.targets[i].item()]))
                                                    )
                                                ])
                            ]
                            for i in range(len(self.targets))]
        self.val_pairs = pos_neg_pairs

    def __len__(self):
        classes_mask = np.where(np.isin(self.zsl_dataset.targets, self.seen_unseen_class_ids))[0]
        return len(self.zsl_dataset.targets[classes_mask])

    def __getitem__(self, index):
        positive_target, negative_target, x_a1, x_v1, x_t1, x_url1, x_a2, x_v2, x_t2, x_url2 = self.negative_random_choose(index)
        
        data = {
            "positive": {"audio": x_a1, "video": x_v1, "text": x_t1, "url": x_url1},
            "negative": {"audio": x_a2, "video": x_v2, "text": x_t2, "url": x_url2},
        }
        target = {
            "positive": positive_target,
            "negative": negative_target,
        }

        return data, target

    def negative_random_choose(self, index):
        if self.dataset_split == "train" or self.dataset_split == "train_val" or self.dataset_split == "train_val_syn" or self.dataset_split == "syn":
            positive_target = self.targets[index].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][index]
            x_v1 = self.data["video"][index]
            x_t1 = self.data["text"][pos_target_index]
            x_url1 = self.data["url"][index]

            negative_target = np.random.choice(list(self.targets_set - set([positive_target])))
            negative_index = np.random.choice(self.target_to_indices[negative_target])
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][negative_index]
            x_v2 = self.data["video"][negative_index]
            x_t2 = self.data["text"][neg_target_index]
            x_url2 = self.data["url"][negative_index]

        elif self.dataset_split == "val" or self.dataset_split == "test":
            positive_target = self.targets[self.val_pairs[index][0]].item()
            pos_target_index = list(self.targets_set).index(positive_target)
            x_a1 = self.data["audio"][self.val_pairs[index][0]]
            x_v1 = self.data["video"][self.val_pairs[index][0]]
            x_t1 = self.data["text"][pos_target_index]
            x_url1 = self.data["url"][self.val_pairs[index][0]]

            negative_target = self.targets[self.val_pairs[index][1]].item()
            neg_target_index = list(self.targets_set).index(negative_target)
            x_a2 = self.data["audio"][self.val_pairs[index][1]]
            x_v2 = self.data["video"][self.val_pairs[index][1]]
            x_t2 = self.data["text"][neg_target_index]
            x_url2 = self.data["url"][self.val_pairs[index][1]]

        else:
            raise AttributeError("Dataset_split has to be either train, val, train_val or test.")
        
        return positive_target, negative_target, x_a1, x_v1, x_t1, x_url1, x_a2, x_v2, x_t2, x_url2