import numpy as np


class SPX_Tester2D:
    DOC_URL = None
    PARAM_HINT = 'gh=9,gw=9' 
    def __init__(self):
        #print('SPX_Tester2D_loaded')
        pass

    def forward(self, **kwargs):
        img = kwargs["img"]

        if img.ndim == 3:
            H, W = img.shape[:2]
        else:
            H, W = img.shape

        gh = kwargs.get('gh',0)
        gw = kwargs.get('gw',0)


        gh, gw = 9, 9
        

        y_coords = np.linspace(0, gh, H, endpoint=False).astype(int)
        x_coords = np.linspace(0, gw, W, endpoint=False).astype(int)

        labels = np.zeros((H, W), dtype=np.int32)

        for i in range(H):
            for j in range(W):
                labels[i, j] = y_coords[i] * gw + x_coords[j] + 1

        return labels
    
class SPX_SLIC2D:
    DOC_URL = "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.slic"
    PARAM_HINT = "n_segments=100, compactness=10, sigma=1"
    def __init__(self):
        from skimage.segmentation import slic
        self.slic = slic
        

    def forward(self, **kwargs):
        img = kwargs.pop("img")
        
        if img is None:
            raise ValueError("Missing required argument: img")
        return self.slic(img, **kwargs)

class SPX_Felzenszwalb2D:
    DOC_URL = "https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb"
    PARAM_HINT = "scale=100, sigma=0.5, min_size=50"
    def __init__(self):
        from skimage.segmentation import felzenszwalb
        self.felzenszwalb = felzenszwalb
        


    def forward(self, **kwargs):
        img = kwargs.pop("img")

        if img is None:
            raise ValueError("Missing required argument: img")
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()

        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        return self.felzenszwalb(img, **kwargs)
        

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
        pass

    @staticmethod
    def instantiate_model(key):
        
        try:
            return globals()[key]()
        
        except:
            raise ValueError(f"{key} not implemented")