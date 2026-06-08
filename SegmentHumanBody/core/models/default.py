"""Default no-op model templates.

These classes are intentionally lightweight placeholders.  They establish the
same registry and family plumbing future model integrations should use without
changing native Segment Editor behavior.
"""


class IdentityModel:
    """No-op model that returns its input image unchanged."""

    PARAM_HINT = "No parameters. Returns the input image unchanged."
    DOC_URL = ""
    REQUIRES_DISTRIBUTIONS: tuple = ()

    def forward(self, **kwargs):
        if "img" not in kwargs:
            raise ValueError("IdentityModel.forward requires 'img'")
        return kwargs["img"]
