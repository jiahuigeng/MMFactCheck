DIRECT_VERACTY_PREDICTION = """You are a professional fact-checking assistant. You are given an IMAGE together with textual CLAIM: {}. 
Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information). Use the following format to provide your response:
 
Prediction: [True or False or NEI(Not Enough Information)] 
"""

# DIRECT_VERACTY_PREDICTION_new = """You are a professional fact-checking assistant. You are given an IMAGE together with textual CLAIM: {}.
# Use the following format to provide your response:
#
# Prediction: [True or False. Please refrain from providing ambiguous assessments.]
# External Knowledge: [Yes or No. "Yes" if you think additional evidence is needed, "No" otherwise.]
# """

COT_VERACTY_PREDICTION = """You are a professional fact-checking assistant. You are given an IMAGE together with textual CLAIM: {}. 
Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information). Use the following format to provide your response:

Prediction: [True or False or NEI(Not Enough Information)]
Explanation: [put your evidence and step-by-step reasoning here]
"""

COT_RUMOR_PREDICTION = """You are a professional fact-checking assistant. You are given an IMAGE together with textual CLAIM: {}.
Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information). Use the following format to provide your response:

Reasoning: [put your step-by-step reasoning here]
Prediction: [True or False or NEI(Not Enough Information)]

Let's think step by step,
"""

ICL1_VERACITY_PREDICTION = """You are a professional fact-checking assistant. Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information). 
Use the following format to provide your response:

Reasoning: [put your step-by-step reasoning here]
Prediction: [True or False or NEI(Not Enough Information)]

For the first IMAGE, claim: {}
Reasoning: {}
Prediction: {}

For the second IMAGE, claim: {}
# Your Response: 
"""

ICL2_VERACITY_PREDICTION = """You are a professional fact-checking assistant. Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information). 
Use the following format to provide your response:

Reasoning: ["put your step-by-step reasoning here"]
Prediction: [True or False or NEI(Not Enough Information)]

For the first IMAGE, claim: {}
Reasoning: {}
Prediction: {}

For the second IMAGE, claim: {}
Reasoning: {}
Prediction: {}

For the third IMAGE, claim: {}
# Your Response: 
"""

RAG_PROMPT = """You are a professional fact-checking assistant. You are first given several retrieved documents. 
{}

Please predict the truth of the multimodal CLAIM: {} regarding to the IMAGE, True, False or NEI(Not Enough Information).

Use the following format to provide your response:

Prediction: [True or False or NEI(Not Enough Information)]
Explanation: [put your evidence and step-by-step reasoning here]
"""

VERACITY_PROMPTS = {
    "direct": DIRECT_VERACTY_PREDICTION,
    "cot": COT_VERACTY_PREDICTION,
    "icl1": ICL1_VERACITY_PREDICTION,
    "icl2": ICL2_VERACITY_PREDICTION,
    "rag": RAG_PROMPT
}

RUMOR_PROMPTS = {
    # "direct": None,
    "cot": None,
    "icl1": None,
    "icl2":None
}

MANIPULATION_PROMPTS = {}

# RAG_PROMPTS = {
#
# }



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



