class BaseModel:
    def __init__(self, widget):
        self.widget = widget

    def on_start(self, *args):
        print("[BaseModel] start")
        self.widget.renderer.start()

    def on_stop(self, *args):
        print("[BaseModel] stop")
        self.widget.renderer.stop()


# ------------------------
# Interactive Model
# ------------------------

class InteractiveModel(BaseModel):
    def on_assign_2d(self, *args):
        print("[Interactive] assign 2D")

    def on_assign_3d(self, *args):
        print("[Interactive] assign 3D")


# ------------------------
# SPX Model (example heavy init)
# ------------------------

class SPXModel(BaseModel):
    def __init__(self, widget):
        super().__init__(widget)
        print("[SPX] initialized (lazy)")

    def on_assign_2d(self, *args):
        print("[SPX] superpixel assign")


# ------------------------
# Auto Model
# ------------------------

class AutoModel(BaseModel):
    def on_start(self, *args):
        print("[Auto] full automatic segmentation")
        self.widget.renderer.start()