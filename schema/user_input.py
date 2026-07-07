from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated
from config.city_tier import tier_1_cities, tier_2_cities



# pydantic model to validate incoming data
class UserInput(BaseModel):

    age: Annotated[int, Field(..., gt=0, lt=120, description= 'Age of the User')]
    weight: Annotated[float, Field(..., gt=0, description= 'Weight of the User')]
    height: Annotated[float, Field(..., gt=0, lt = 2.5, description= 'height of the User')]
    income_lpa: Annotated[float, Field(..., description= 'Annual salary of the User in lpa')]
    smoker: Annotated[bool, Field(..., description= 'Is User a smoker')]
    city: Annotated[str, Field(..., description= 'The city that the user belongs to')]
    occupation: Annotated[Literal[ "Teacher", "Business Owner", "Student", "Retired","Doctor", 
    "Software Engineer", "Sales Executive", "Banker"], Field(..., description= 'Occupation of the User')]

    
     
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
        return self.weight/ ((self.height/100)**2)


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
        elif self.age <45:
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