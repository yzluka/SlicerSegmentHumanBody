class BaseModelFamily:
    VARIANTS = ['None']
    def __init__(self, widget):
        self.widget = widget
        self.variant = None
        print(f"INIT CALLED: {type(self).__name__}")

    def on_confirm_model_selection(self):
        variant = self.widget.ui.modelVariantDropdown.currentText

        if not variant:
            print("[Confirm] No variant selected")
            return

        self.variant = variant  # 🔥 sync here

        print(f"[Confirm] {type(self).__name__} → {self.variant}")

        self._check_dependencies()
        self._load_model()
    def _check_dependencies(self):
        print(f"[Dependencies] Checking for {self.variant}")

    def _load_model(self):
        print(f"[Load] Loading model {self.variant}")

# ------------------------
# Interactive Model
# ------------------------

class InteractiveModelFamily(BaseModelFamily):
    VARIANTS = ['SAM-VIT-H','SAM-ViT-L', 'SAM-ViT-B', 'sam2_hiera_l','sam2_hiera_b+','sam2_hiera_s', 'sam2_hiera_t']
    def __init__(self, widget):
        super().__init__(widget)
    
    def on_assign_2d(self, *args):
        print("[Interactive] assign 2D")

    def on_assign_3d(self, *args):
        print("[Interactive] assign 3D")

    def on_start(self, *args):
        print("[Interactive] start")
        self.widget.renderer.start()

    def on_stop(self, *args):
        print("[Interactive] stop")
        self.widget.renderer.stop()
    def on_go_to_markups(self, *args):
        pass



class SPXModelFamily(BaseModelFamily):
    VARIANTS = ['SLIC','Felzenszwalb']
    def __init__(self, widget):
        super().__init__(widget)

    def on_assign_2d(self, *args):
        print("[SPX] superpixel assign")
    
    def on_propagate(self, *args):
        print("[SPX] propagate")



class AutoModelFamily(BaseModelFamily):
    VARIANTS = ['BreastCT','PE_SEG']
    
    def __init__(self, widget):
        super().__init__(widget)

    def on_start(self, *args):
        print("[Auto] full automatic segmentation")
        self.widget.renderer.start()
    