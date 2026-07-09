from pydantic import BaseModel

class RoleUpdate(BaseModel):
    role: str  # e.g. "admin" or "user"

class StatusUpdate(BaseModel):
    is_active: bool