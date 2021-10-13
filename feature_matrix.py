from pydantic import BaseModel
#create class that describes the feature matrix
class features(BaseModel):
    radius_mean=float
    texture_mean=float
    perimeter_mean=float
    area_mean=float
    smoothness_mean=float
    compactness_mean=float
    concavity_mean=float
    concave_points_mean=float
    symmetry_mean=float
    fractal_dimension_mean=float
