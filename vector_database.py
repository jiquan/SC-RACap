import pickle

import faiss
import torch
from torch.utils.data import DataLoader

from dataset.coco_dataset import COCODataset
from dataset.miccai_dataset import MICCAIDataset
from models.eva_vit import create_eva_vit_g
from models.scracap import SCRACap, SCRACAP_IMG


def load_vector(ext_path='ext_data/ext_memory_miccai_new.pkl'):
    with open(ext_path, 'rb') as f:
        ext_base_img, ext_base_img_id, ext_base_triplet = pickle.load(f)
        print(ext_base_img.shape, len(ext_base_img_id))
        feature_library_cpu = ext_base_img.cpu().numpy()
        faiss.normalize_L2(feature_library_cpu)
        feat_index = faiss.IndexFlatIP(feature_library_cpu.shape[1])
        feat_index.add(feature_library_cpu)
        print(f"loaded external base image")


def collate_fn(batch):
    images = [item['image'] for item in batch]
    text_inputs = [item['text_input'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    instruments = [item['instrument'] for item in batch]
    labels = [item['labels'] for item in batch]
    shorts = [item['short'] for item in batch]

    return {
        'image': torch.stack(images, dim=0),
        'text_input': text_inputs,
        'image_id': image_ids,
        'instrument': instruments,
        'labels': labels,
        'short': shorts
    }


def make_vector():
    data_root = './data/miccai_coco'
    dataset = MICCAIDataset(data_root=data_root)
    model = SCRACAP_IMG(
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
    )
    device = torch.device("cuda:0")
    batch_size = 6
    sampler = None
    model = model.to(device)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn)
    embeddings = []
    ext_base_img_id = []
    ext_base_triplet = []
    for idx, samples in enumerate(train_dataloader):
        samples['image'] = samples['image'].to(device)
        img_embedding = model.img_vector(samples['image']).detach()
        embeddings.append(img_embedding)
        ext_base_img_id.extend(samples['labels'])
        ext_base_triplet.extend(samples['short'])
        # if idx > 2:
        #     break

    ext_base_img = torch.cat(embeddings, dim=0)
    with open('ext_data/ext_memory_miccai_new.pkl', 'wb') as f:
        pickle.dump((ext_base_img, ext_base_img_id, ext_base_triplet), f)


if __name__ == '__main__':
    make_vector()
    # load_vector()