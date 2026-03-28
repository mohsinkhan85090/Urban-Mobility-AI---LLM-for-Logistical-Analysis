from pydantic import BaseModel, Field


class RouteInput(BaseModel):
    pickup_zone: str = Field(..., description="Pickup zone name")
    dropoff_zone: str = Field(..., description="Dropoff zone name")


class UrbanTripPlannerInput(BaseModel):
    pickup_zone: str = Field(..., description="Pickup zone name")
    dropoff_zone: str = Field(..., description="Dropoff zone name")
