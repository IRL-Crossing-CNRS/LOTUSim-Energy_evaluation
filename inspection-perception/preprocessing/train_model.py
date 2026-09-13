from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-seg.pt")


    model.train(
        data="data.yaml",

        epochs=100,
        patience=25,
        imgsz=832,
        batch=8,
        device=0,

        optimizer="SGD",
        lr0=0.00777,
        lrf=0.00705,
        momentum=0.8515,
        weight_decay=0.0005,
        warmup_epochs=2.52,
        warmup_momentum=0.7246,

        box=7.6411,
        cls=0.5715,
        dfl=1.4230,

        cos_lr=False,
        close_mosaic=10,
        mask_ratio=4,
        save_period=5,

        seed=0,
        deterministic=True,
        name="v1_img832",
    )


if __name__ == "__main__":
      main()
