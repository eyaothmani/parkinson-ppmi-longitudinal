"""État partagé du service : modèle chargé, sa version et les compteurs de monitoring."""
from ..monitoring.service import ServiceMetrics

state = {"predictor": None, "state_model": None, "factors": None, "rag": None, "info": {"version": "aucun"},
         "metrics": ServiceMetrics()}
