from datasets import load_dataset
from utils import pad_to, get_img_and_bbox, draw_bbox_and_save

ds_train = load_dataset("CUHK-CSE/wider_face", split="train")
ds_val = load_dataset("CUHK-CSE/wider_face", split="validation")
ds_test = load_dataset("CUHK-CSE/wider_face", split="test")


def main():
    for i, t in enumerate(ds_train):
        img, bbox = get_img_and_bbox(t)
        new_img, new_bbox = pad_to(img, bbox)

        out_path = f"outputs/img_{i}.jpg"
        draw_bbox_and_save(new_img, new_bbox, out_path)

        if i > 10:
            break


if __name__ == "__main__":
    main()
