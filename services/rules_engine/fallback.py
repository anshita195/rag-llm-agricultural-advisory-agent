"""
Safety fallback rules engine for AgriSage
Provides deterministic responses when LLM confidence is low or for critical decisions
"""

def irrigation_rule(soil_moisture, precip_prob):
    """
    Irrigation decision rule based on soil moisture and precipitation probability
    
    Args:
        soil_moisture: Soil moisture percentage (0-100)
        precip_prob: Precipitation probability percentage (0-100)
    
    Returns:
        dict: {"action": str, "advice": str, "confidence": float}
    """
    if soil_moisture is None or precip_prob is None:
        return {
            "action": "insufficient_data",
            "advice": "Insufficient soil moisture or weather data. Please check your local weather station or agricultural extension officer for current conditions.",
            "confidence": 0.3
        }
    
    if soil_moisture < 30 and precip_prob < 30:
        return {
            "action": "irrigate_now",
            "advice": "Soil moisture low and low chance of rain — irrigate in the evening.",
            "confidence": 0.90
        }
    
    if precip_prob > 70:
        return {
            "action": "delay_irrigation",
            "advice": "Heavy rain likely — delay irrigation.",
            "confidence": 0.95
        }
    
    return {
        "action": "consult",
        "advice": "Insufficient confidence — consult extension agent.",
        "confidence": 0.40
    }

def fertilizer_rule(crop, growth_stage, soil_n, soil_p, soil_k):
    """
    Basic fertilizer recommendation rule
    
    Args:
        crop: Crop name
        growth_stage: Growth stage (seedling, vegetative, flowering, maturity)
        soil_n, soil_p, soil_k: Soil nutrient levels
    
    Returns:
        dict: {"action": str, "advice": str, "confidence": float}
    """
    if any(x is None for x in [soil_n, soil_p, soil_k]):
        return {
            "action": "soil_test",
            "advice": "Soil nutrient data not available. Please get your soil tested at a local agricultural laboratory for accurate fertilizer recommendations.",
            "confidence": 0.85
        }
    
    # Basic NPK deficiency rules
    deficiencies = []
    if soil_n < 280:  # kg/ha
        deficiencies.append("Nitrogen")
    if soil_p < 11:   # kg/ha
        deficiencies.append("Phosphorus")
    if soil_k < 120:  # kg/ha
        deficiencies.append("Potassium")
    
    if deficiencies:
        return {
            "action": "consult_expert",
            "advice": f"Soil shows {', '.join(deficiencies)} deficiency. Consult agricultural officer for specific fertilizer recommendations.",
            "confidence": 0.75
        }
    
    return {
        "action": "balanced_fertilizer",
        "advice": "Soil nutrients appear adequate. Use balanced fertilizer as per crop requirements.",
        "confidence": 0.70
    }

def pest_disease_rule(symptoms, crop):
    """
    Pest and disease identification rule - always escalates for safety
    
    Args:
        symptoms: Description of symptoms
        crop: Crop name
    
    Returns:
        dict: {"action": str, "advice": str, "confidence": float}
    """
    return {
        "action": "escalate",
        "advice": "Pest and disease diagnosis requires expert examination. Contact your nearest Krishi Vigyan Kendra or agricultural extension officer immediately.",
        "confidence": 1.0
    }

def market_timing_rule(commodity, current_price, historical_avg):
    """
    Market timing advice rule
    
    Args:
        commodity: Commodity name
        current_price: Current market price
        historical_avg: Historical average price
    
    Returns:
        dict: {"action": str, "advice": str, "confidence": float}
    """
    if current_price is None or historical_avg is None:
        return {
            "action": "check_market",
            "advice": "Check current market prices at nearby mandis before selling.",
            "confidence": 0.60
        }
    
    price_ratio = current_price / historical_avg
    
    if price_ratio > 1.15:  # 15% above average
        return {
            "action": "sell_now",
            "advice": f"Current price is {((price_ratio-1)*100):.1f}% above average. Good time to sell.",
            "confidence": 0.80
        }
    
    if price_ratio < 0.85:  # 15% below average
        return {
            "action": "wait_or_store",
            "advice": f"Current price is {((1-price_ratio)*100):.1f}% below average. Consider waiting if you can store safely.",
            "confidence": 0.75
        }
    
    return {
        "action": "market_normal",
        "advice": "Prices are near average. Sell based on your immediate needs.",
        "confidence": 0.65
    }

def safety_check(question_text):
    """
    Check if question contains risky keywords that require escalation
    
    Args:
        question_text: User's question text
    
    Returns:
        bool: True if question should be escalated
    """
    risky_keywords = [
        'pesticide', 'insecticide', 'fungicide', 'herbicide',
        'dose', 'dosage', 'ppm', 'spray', 'chemical',
        'poison', 'toxic', 'ml/acre', 'gm/acre',
        'concentration', 'dilution'
    ]
    
    question_lower = question_text.lower()
    return any(keyword in question_lower for keyword in risky_keywords)

def get_fallback_response(question, context=None):
    """
    Main fallback function that routes to appropriate rule
    
    Args:
        question: User's question
        context: Optional context data
    
    Returns:
        dict: Fallback response with action, advice, and confidence
    """
    question_lower = question.lower()
    
    # Safety check first
    if safety_check(question):
        return {
            "action": "escalate",
            "advice": "This question involves chemicals or dosages. Please consult your local agricultural extension officer or Krishi Vigyan Kendra for safe recommendations.",
            "confidence": 1.0,
            "escalate": True
        }
    
    # Route to specific rules based on question content
    if any(word in question_lower for word in ['irrigat', 'water', 'moisture']):
        soil_moisture = context.get('soil_moisture') if context else None
        precip_prob = context.get('precip_prob') if context else None
        result = irrigation_rule(soil_moisture, precip_prob)
        
    elif any(word in question_lower for word in ['fertiliz', 'nutrient', 'npk']):
        crop = context.get('crop') if context else None
        growth_stage = context.get('growth_stage') if context else None
        soil_n = context.get('soil_n') if context else None
        soil_p = context.get('soil_p') if context else None
        soil_k = context.get('soil_k') if context else None
        result = fertilizer_rule(crop, growth_stage, soil_n, soil_p, soil_k)
        
    elif any(word in question_lower for word in ['pest', 'disease', 'insect', 'fungus', 'virus']):
        result = pest_disease_rule(question, context.get('crop') if context else None)
        
    elif any(word in question_lower for word in ['price', 'market', 'sell', 'mandi']):
        commodity = context.get('commodity') if context else None
        current_price = context.get('current_price') if context else None
        historical_avg = context.get('historical_avg') if context else None
        result = market_timing_rule(commodity, current_price, historical_avg)
        
    else:
        # Generic fallback
        result = {
            "action": "consult",
            "advice": "For specific agricultural advice, please consult your local agricultural extension officer or visit the nearest Krishi Vigyan Kendra.",
            "confidence": 0.50
        }
    
    # Add escalate flag if confidence is very low
    if result["confidence"] < 0.4:
        result["escalate"] = True
    
    return result
