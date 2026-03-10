"""
Personality JSON template and LLM prompt builder for character generation.
"""

import json

PERSONALITY_TEMPLATE = {
    "identity": {
        "age": 0,
        "gender": "",
        "location": {
            "current_city": "",
            "current_country": "",
            "years_at_current_location": 0
        }
    },
    "biological_factors": {
        "genetics": {
            "risk_tolerance_genetic_component": 0.0,
            "impulsivity_genetic_component": 0.0
        },
        "physical_health": {
            "sleep_quality": 0.0,
            "average_sleep_hours": 0.0,
            "diet_quality": 0.0,
            "energy_level": 0.0
        }
    },
    "personality": {
        "big_five": {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0
        },
        "risk_tolerance": 0.0,
        "impulsivity": 0.0,
        "optimism": 0.0
    },
    "cognitive_factors": {
        "intelligence": {
            "iq_estimate": 0,
            "working_memory_capacity": 0,
            "verbal_ability": 0.0,
            "numerical_reasoning": 0.0
        },
        "thinking_styles": {
            "analytical_thinking": 0.0,
            "creative_thinking": 0.0
        }
    },
    "emotional_factors": {
        "current_emotional_state": {
            "stress_level": 0.0,
            "happiness_level": 0.0,
            "emotional_volatility": 0.0
        }
    },
    "economic_factors": {
        "current_financial_situation": {
            "annual_income_usd": 0,
            "savings_usd": 0,
            "debt_usd": 0,
            "financial_stress_level": 0.0
        }
    },
    "psychological_patterns": {
        "motivation": {
            "achievement_motivation": 0.0,
            "autonomy_need": 0.0
        },
        "coping_mechanisms": {
            "emotion_focused_coping": 0.0,
            "avoidance_coping": 0.0
        },
        "emotional_regulation": {
            "regulation_capacity": 0.0,
            "preferred_strategies": [],
            "suppression_tendency": 0.0,
            "rumination_tendency": 0.0
        },
        "attachment_style": {
            "primary_style": "",
            "anxious_component": 0.0,
            "avoidant_component": 0.0,
            "trust_baseline": 0.0
        }
    },
    "past_experiences": {
        "childhood": {
            "socioeconomic_status": "",
            "parenting_style": "",
            "childhood_trauma": "",
            "significant_early_experiences": [],
            "early_attachment_security": 0.0
        },
        "education": {
            "highest_degree": "",
            "quality_of_education": 0.0,
            "learning_disabilities": "",
            "academic_achievement": ""
        },
        "significant_life_events": [],
        "relationship_history": {
            "number_of_serious_relationships": 0
        }
    },
    "memory_and_learning": {
        "memory_systems": {
            "episodic_memory_quality": 0.0,
            "semantic_memory_quality": 0.0,
            "working_memory_capacity": 0
        }
    },
    "existential_and_meaning": {
        "meaning_in_life": 0.0,
        "existential_anxiety": 0.0,
        "life_satisfaction": 0.0
    },
    "decision_making_competencies": {
        "decision_confidence": {
            "general_confidence": 0.0,
            "overconfidence_tendency": 0.0
        }
    }
}

GENERATION_SYSTEM_PROMPT = """You are a character designer. You generate detailed character personality profiles as JSON.

RULES:
- Output ONLY valid JSON. No markdown fences, no explanation, no commentary.
- Match the EXACT structure of the template provided. Do not add or remove any keys.
- All numeric values between 0.0 and 1.0 are normalized scales (0=low, 1=high) unless otherwise clear from context (e.g. age, income, IQ).
- stress_level and happiness_level are on a 1-10 scale.
- sleep_quality, diet_quality, energy_level are on a 1-5 scale.
- quality_of_education is on a 1-10 scale.
- life_satisfaction is on a 1-10 scale.
- Be creative but internally consistent with the character's established identity.
- significant_life_events should be an array of objects with: event, age, impact, lasting_effects.
- preferred_strategies should be an array of 2-4 strings.
- significant_early_experiences should be an array of 1-3 strings.
- The character should feel like a real person with a coherent backstory."""


def build_generation_prompt(
    agent_name: str,
    agent_title: str,
    agent_specialty: str,
    system_prompt: str,
    user_hint: str = "",
) -> str:
    """Build the user message for personality generation."""
    template_str = json.dumps(PERSONALITY_TEMPLATE, indent=2)

    parts = [
        f"Generate a detailed character personality JSON for this person:\n",
        f"Name: {agent_name}",
        f"Title: {agent_title}",
        f"Specialty: {agent_specialty}",
        f"\nTheir established character description:\n{system_prompt}",
        f"\nFill in this exact JSON template with values that fit this character:\n{template_str}",
    ]

    if user_hint:
        parts.append(f"\nAdditional instructions: {user_hint}")

    parts.append("\nRespond with ONLY the filled JSON, nothing else.")
    return "\n".join(parts)
