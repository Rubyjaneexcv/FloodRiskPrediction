"""Page views for the Flood Risk Prediction System."""

from views import (
    about,
    analysis,
    dashboard,
    dataset,
    history,
    mapping,
    prediction,
)

PAGES = {
    "dashboard": dashboard.render,
    "prediction": prediction.render,
    "map": mapping.render,
    "analysis": analysis.render,
    "history": history.render,
    "dataset": dataset.render,
    "about": about.render,
}

__all__ = ["PAGES", "dashboard", "prediction", "mapping", "analysis",
           "history", "dataset", "about"]
