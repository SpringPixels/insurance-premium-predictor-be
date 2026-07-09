from pydantic import BaseModel

class SegmentResponse(BaseModel):
    cluster_id: int
    segment_label: str
    recommended_activity: str
    goal: str

class TrainSegmentsResponse(BaseModel):
    n_users: int
    cluster_sizes: dict
    label_map: dict