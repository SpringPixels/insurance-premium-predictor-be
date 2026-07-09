import asyncio
import random
from config.database import get_db
from ml_model.db_models import PredictionInput, PredictionLog

AGE_GROUPS = ["young", "adult", "middle_aged", "senior"]
LIFESTYLE_RISKS = ["low", "medium", "high"]
OCCUPATIONS = ["salaried", "self_employed", "business_owner", "student", "retired"]
CATEGORIES = ["Low", "Medium", "High"]


def bmi_from(weight, height_cm):
    h_m = height_cm / 100
    return round(weight / (h_m ** 2), 1)


def age_group_from(age):
    if age < 30: return "young"
    if age < 45: return "adult"
    if age < 60: return "middle_aged"
    return "senior"


async def seed(n=60):
    async for db in get_db():
        for _ in range(n):
            lifestyle_risk = random.choice(LIFESTYLE_RISKS)
            age = random.randint(18, 75)
            height = random.randint(150, 195)

            if lifestyle_risk == "high":
                weight = round(random.uniform(85, 130), 1)
            elif lifestyle_risk == "medium":
                weight = round(random.uniform(65, 90), 1)
            else:
                weight = round(random.uniform(50, 75), 1)

            pi = PredictionInput(
                age=age,
                weight=weight,
                height=height,
                is_smoker=random.random() < 0.2,
                occupation=random.choice(OCCUPATIONS),
                income_lpa=round(random.uniform(3, 40), 1),
                city=random.choice(["Mumbai", "Delhi", "Pune", "Nagpur", "Nashik"]),
            )
            db.add(pi)
            await db.flush()

            pl = PredictionLog(
                input_id=pi.id,
                income_lpa=pi.income_lpa,
                occupation=pi.occupation,
                bmi=bmi_from(weight, height),
                age_group=age_group_from(age),
                lifestyle_risk=lifestyle_risk,
                city_tier=random.choice([1, 2, 3]),
                predicted_category=random.choice(CATEGORIES),
                predicted_premium=round(random.uniform(5000, 50000), 2),
            )
            db.add(pl)

        await db.commit()
        print(f"Seeded {n} PredictionInput + PredictionLog rows")
        break


if __name__ == "__main__":
    asyncio.run(seed(60))