from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated, Dict
from config.city_tier import tier_1_cities, tier_2_cities

class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description='Age of the User')]
    weight: Annotated[float, Field(..., gt=0, description='Weight of the User in kg')]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description='Height of the User in meters')]
    income_lpa: Annotated[float, Field(..., description='Annual salary of the User in lpa')]
    smoker: Annotated[bool, Field(..., description='Is User a smoker')]
    city: Annotated[str, Field(..., description='The city that the user belongs to')]
    occupation: Annotated[Literal["Teacher", "Business Owner", "Student", "Retired", "Doctor",
                                  "Software Engineer", "Sales Executive", "Banker"], Field(..., description='Occupation of the User')]

    @field_validator('city', mode='before')
    @classmethod
    def normalize_city(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().title()
        return v

    @field_validator('occupation', mode='before')
    @classmethod
    def normalize_occupation(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().title()
        return v

    @computed_field
    @property
    def bmi(self) -> float:
        # Fixed bug: Height is already in meters based on validation (lt=2.5)
        # Old buggy code: self.weight / ((self.height/100)**2)
        return self.weight / (self.height ** 2)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker:
            return "medium"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        else:
            return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3

class PredictionResponse(BaseModel):
    predicted_category: str = Field(
        ...,
        description="The predicted insurance premium category",
        example="High"
    )
    confidence: float = Field(
        ...,
        description="Model's confidence score for the predicted class (range: 0 to 1)",
        example=0.8432
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all possible classes",
        example={"Low": 0.01, "Medium": 0.15, "High": 0.60}
    )
    # the exact dictionary output from predict_and_explain can be mapped to this, but we'll accept extra fields for metadata and shap if needed
    model_metadata: dict | None = None
    prediction_results: dict | None = None
    explainable_ai: dict | None = None
    predicted_premium: int = Field(
        ...,
        description="The calculated premium amount based on the category",
        example=18000
    )

class ScenarioResult(BaseModel):
    model_metadata: dict
    prediction_results: dict
    explainable_ai: dict
    predicted_premium: int

class CompareResponse(BaseModel):
    scenario_a: ScenarioResult
    scenario_b: ScenarioResult
    difference: int
