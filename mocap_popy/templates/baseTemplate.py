"""!
    Base Template Class
    ===============================

    All templates should inherit from this class. 
    This enforces a consistent interface for exported json templates adn other scripts (i.e. template_loader).

    @author C. McCarthy
"""

from pydantic import BaseModel, Field


class BaseTemplate(BaseModel):
    name: str = Field(..., description="The name of the template.")
    parent_models: list[str] = Field(
        default_factory=list,
        description="List of models (labeling skeleton templates) that contain this template.",
    )
