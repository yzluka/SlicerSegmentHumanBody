import numpy as np


class SPX_Tester:
    def __init__(self):
        print('SPX_Tester_loaded')

    def forward(self, *args, **kwargs):
        img = kwargs["img"]

        if img.ndim == 3:
            H, W = img.shape[:2]
        else:
            H, W = img.shape

        print("[SPX] H, W =", H, W)

        gh, gw = 9, 9

        y_coords = np.linspace(0, gh, H, endpoint=False).astype(int)
        x_coords = np.linspace(0, gw, W, endpoint=False).astype(int)

        labels = np.zeros((H, W), dtype=np.int32)

        for i in range(H):
            for j in range(W):
                labels[i, j] = y_coords[i] * gw + x_coords[j] + 1

        print("[SPX] unique labels:", len(np.unique(labels)))

        return labels


class ModelRegistry:
    model_cache = {}

    @staticmethod
    def get_model(key):
        if key not in ModelRegistry.model_cache:
            ModelRegistry.check_dependencies(key)
            model = ModelRegistry.instantiate_model(key)
            ModelRegistry.model_cache[key] = model
            return model

        return ModelRegistry.model_cache[key]

    @staticmethod
    def check_dependencies(key):
        print(f"[Dependencies] Checking for {key}")

    @staticmethod
    def instantiate_model(key):
        print(f"[Instantiation] Fetching model for {key}")

        if key == 'SPX_Tester':
            return SPX_Tester()

        raise ValueError(f"{key} not implemented")