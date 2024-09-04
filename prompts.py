DIRECT_VERACTY_PREDICTION = """You are a professional fact-checking assistant. You are given an IMAGE together with textual CLAIM: {}. 
Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information).

# Your Response:
Prediction: [True or False or NEI(Not Enough Information)] 
"""

COT_VERACTY_PREDICTION = """You are a professional fact-checking assistant. You are given an IMAGE together with textual CLAIM: {}. 
Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information).
Prediction: [True or False or NEI(Not Enough Information)] 

# Your Response:
Let's think step by step.
Reasoning: ["put your step-by-step reasoning here"]
Prediction: [True or False or NEI(Not Enough Information)]
"""

ICL_VERACITY_PREDICTION = """
"""

VERACITY_PROMPTS = {
    "direct": DIRECT_VERACTY_PREDICTION,
    "cot": COT_VERACTY_PREDICTION,
    "icl": ICL_VERACITY_PREDICTION
}

MANIPULATION_PROMPTS = {}

DEMONSTRATIONS = {
    "0": {
        "claim": "",
        "image_id": "",
        "reasoning": "",
        "label": ""
    },
    "1": {
        "claim": "",
        "image_id": "",
        "reasoning": "",
        "label": ""
    },
    "2": {
        "claim": "",
        "image_id": "",
        "reasoning": "",
        "label": ""
    },
    "3": {
        "claim": "",
        "image_id": "",
        "reasoning": "",
        "label": ""
    },

}



