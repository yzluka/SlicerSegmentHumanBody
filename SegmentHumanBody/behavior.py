import slicer


class BaseBehavior:
    def __init__(self, widget):
        self.widget = widget

# ------------------------
# Example Behaviors
# ------------------------

class InteractiveBehavior(BaseBehavior):
    def on_assign_2d(self, *args):
        print("[Interactive] assign 2D")

    def on_assign_3d(self, *args):
        print("[Interactive] assign 3D")


class SPXBehavior(BaseBehavior):
    def on_assign_2d(self, *args):
        print("[SPX] assign using superpixels")


class AutoBehavior(BaseBehavior):
    def on_start(self, *args):
        print("[Auto] full automatic segmentation")
        self.widget.renderer.start()