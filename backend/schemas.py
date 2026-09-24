"""Request / response models for the prediction API (input validation with Pydantic).

Only information known BEFORE the call is accepted. `duration` is deliberately not a field.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Job = Literal["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
              "self-employed", "services", "student", "technician", "unemployed", "unknown"]
Marital = Literal["divorced", "married", "single", "unknown"]
Education = Literal["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                    "professional.course", "university.degree", "unknown"]
YesNoUnknown = Literal["yes", "no", "unknown"]
Contact = Literal["cellular", "telephone"]
Month = Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
Weekday = Literal["mon", "tue", "wed", "thu", "fri"]
Poutcome = Literal["failure", "nonexistent", "success"]

# API field name -> column name used when the model was trained
COLUMN_MAP = {"emp_var_rate": "emp.var.rate", "cons_price_idx": "cons.price.idx",
              "cons_conf_idx": "cons.conf.idx", "euribor3m": "euribor3m", "nr_employed": "nr.employed"}


class ClientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")          # unknown fields (e.g. duration) are rejected

    age: int = Field(..., ge=17, le=100, description="Client age in years")
    job: Job
    marital: Marital
    education: Education
    default: YesNoUnknown = Field(..., description="Has credit in default?")
    housing: YesNoUnknown = Field(..., description="Has a housing loan?")
    loan: YesNoUnknown = Field(..., description="Has a personal loan?")
    contact: Contact = Field(..., description="Contact channel")
    month: Month = Field(..., description="Month of the contact")
    day_of_week: Weekday
    campaign: int = Field(..., ge=1, le=60, description="Number of contacts in this campaign (as recorded in the dataset)")
    pdays: int = Field(..., ge=0, le=999, description="Days since previous campaign contact; 999 = never contacted")
    previous: int = Field(..., ge=0, le=20, description="Contacts before this campaign")
    poutcome: Poutcome = Field(..., description="Outcome of the previous campaign")
    emp_var_rate: float = Field(..., ge=-5, le=3, description="Employment variation rate")
    cons_price_idx: float = Field(..., ge=90, le=96, description="Consumer price index")
    cons_conf_idx: float = Field(..., ge=-60, le=-20, description="Consumer confidence index")
    euribor3m: float = Field(..., ge=0, le=6, description="3-month Euribor rate")
    nr_employed: float = Field(..., ge=4700, le=5400, description="Number of employed (thousands)")


class PredictionResponse(BaseModel):
    probability: float = Field(..., description="Estimated probability that the client subscribes")
    predicted_class: Literal["yes", "no"]
    recommendation: Literal["Call", "Do not call"]
    priority: Literal["High", "Medium", "Low"]
    threshold: float
    model_name: str
    note: str = "Decision support only. Check contact preferences and opt-outs before calling."
