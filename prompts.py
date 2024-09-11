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

Prediction: [True or False or NEI(Not Enough Information)]
Explanation: [put your evidence and step-by-step reasoning here]

For the first IMAGE, claim: {}
Prediction: {}
Explanation: {}

For the second IMAGE, claim: {}
"""

ICL2_VERACITY_PREDICTION = """You are a professional fact-checking assistant. Please predict the truth of the multimodal claim, True, False or NEI(Not Enough Information). 
Use the following format to provide your response:

Prediction: [True or False or NEI(Not Enough Information)]
Explanation: [put your evidence and step-by-step reasoning here]

For the first IMAGE, claim: {}
Prediction: {}
Explanation: {}

For the second IMAGE, claim: {}
Prediction: {}
Explanation: {}

For the third IMAGE, claim: {}
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



MANIPULATION_PROMPTS = {}



DEMONSTRATIONS = {
    "0": {
        "claim": "A photograph shows a man surfing with a whale.",
        "image_id": "dataset/demonstration/07803-proof-03-surferwhale_fb.jpg",
        "reasoning": "The image you've provided appears to be digitally manipulated. While surfing and whales are both associated with the ocean, the scale and proximity of the whale to the surfer in this photograph are not consistent with typical whale behavior or the physics of ocean waves. Whales are generally not found surfing in waves, and their size compared to the surfer seems disproportionate. Additionally, the water disturbance caused by a whale of that size would likely be much more significant than what is depicted here. The lack of any news coverage or scientific documentation of such an extraordinary event further suggests that this image is not an accurate representation of reality. ",
        "label": "False"
    },
    "1": {
        "claim": "A photograph shows Morgan Freeman on the children's show 'The Electric Company.",
        "image_id": "dataset/demonstration/12809-proof-02-morgan-freeman.jpg",
        "reasoning": """The man in the photograph is indeed Morgan Freeman. He began his career with appearances in several children’s educational programs, including "The Electric Company," which aired from 1971 to 1977. In "The Electric Company," Freeman played several characters, including the iconic Easy Reader, which is likely the character depicted in this photograph, as indicated by his attire and the overall style of the set. This aligns with the known history of Freeman's roles and the time period in which the show was popular.""",
        "label": "True"
    },
    "2": {
        "claim": "A photograph shows rays of lights in the shape of a cross during the August 2017 eclipse.",
        "image_id": "dataset/demonstration/04247-proof-01-2017_eclipse_cross_faux_fb.jpg",
        "reasoning": """The claim that a photograph shows rays of light arranged in a perfect cross shape during the August 2017 eclipse is not supported by astronomical events or the physics of how light behaves during an eclipse. During a total solar eclipse, phenomena such as Baily's beads or the diamond ring effect may occur, which can result in spectacular visual displays as the moon covers or uncovers the sun. However, a cross shape made of light rays is not a typical occurrence and could instead be a sign of digital manipulation or an artifact of the photography process rather than a natural event. A natural cross shape in the sky during an eclipse would be an extraordinary claim requiring substantial evidence, which is not typically reported or documented by reliable sources in astronomy.""",
        "label": "False"
    },
    "3": {
        "claim": "A photograph captured in mid-June 2021 showed the Robertson Draw Fire burning a mountainside outside of Red Lodge, Montana.",
        "image_id": "dataset/demonstration/14332-proof-10-Copy-of-Rating-Overlay-FEATURED-IMG-1-2.jpg",
        "reasoning": """The image provided shows a large fire on a mountainside in the background with a clear foreground of a street with buildings, vehicles, and an American flag, characteristic of a small town setting. Given the description of the photo matching the image and knowing that wildfires in Montana around mid-June of 2021, including the Robertson Draw Fire which occurred near Red Lodge, were reported in various news outlets, it is reasonable to conclude that the statement is true.""",
        "label": "True"
    },
    "4": {
        "claim": "A photograph depicts a glamorous 'young Harriet Tubman.",
        "image_id": "dataset/demonstration/04239-proof-07-young-harriet-tubman2.jpg",
        "reasoning": """The photograph on the left, which is sometimes claimed to depict a young Harriet Tubman, is actually a misidentified image. The woman in the photograph was actually not Harriet Tubman. Tubman was born in the early 1820s, and there are only a few authenticated photographs of her, with the most famous images showing her as an older woman (as seen on the right). Debunking the myth has involved comparisons of facial features with known photographs of Harriet Tubman as well as historical research into the clothing and photography techniques of the era.""",
        "label": "False"
    },
    "5": {
        "claim": "Photographs show thieving birds stealing quarters from a car wash.",
        "image_id": "dataset/demonstration/12372-proof-04-birdcoin.jpg",
        "reasoning": """The photograph you provided shows a bird sticking its beak into a coin dispenser, which is a behavior consistent with the claim that birds could be stealing quarters. Birds are known for their intelligence and adaptability, and there are documented cases of birds such as crows using tools and solving problems. It is plausible that a bird learned the association between the coin dispenser and food (since some people might drop food coins there), or they might be attracted to the shiny coins for other reasons such as collecting materials or simply curiosity. Without further context or multiple sources, it's difficult to confirm the intentions of the bird or the frequency of such thievery, but the photo suggests the possibility.""",
        "label": "True"
    },
    "6": {
        "claim": "Photograph shows Hillary Clinton shaking hands with Osama bin Laden.",
        "image_id": "dataset/demonstration/04124-proof-10-hillary_clinton_shaking_hands_osama_bin_laden_faux_fb.jpg",
        "reasoning": """The claim that the photograph shows Hillary Clinton shaking hands with Osama bin Laden is a known piece of misinformation. Such a meeting between a high-profile American political figure and the then-leader of Al-Qaeda would have been highly unlikely and well-documented had it happened. Furthermore, the nature of this image, likely appearing on the internet, suggests that it could have been digitally manipulated or a complete fabrication. Without a credible source or confirmation from trustworthy news sources, such a claim should be considered false.""",
        "label": "False"
    },
    "7": {
        "claim": "A photograph shows the dramatic differences in two boys who were exposed to the same smallpox source.",
        "image_id": "dataset/demonstration/14195-proof-05-vaccinattions-1.jpg",
        "reasoning": "The photograph shown is a classic depiction often used to demonstrate the efficacy of vaccines. The boy on the top appears to have only a few lesions and is in relatively good health, whereas the boy on the bottom is covered with extensive smallpox lesions. This stark contrast is historically used to illustrate the effects of smallpox vaccination. The vaccinated individual (presumably the boy on the top) displays milder symptoms after exposure to the virus, while the unvaccinated individual (presumably the boy on the bottom) shows the full severity of the disease. This image is commonly used in educational materials to highlight the success of vaccination in preventing severe disease.",
        "label": "True"
    },
}


COT_AUTHEN_PREDICTION = """You are an advanced AI model trained to assess images for authenticity. Your task is to analyze the given image and determine whether it is real or fake. Provide a clear and concise response with your prediction and an explanation supporting your decision.

Please output your response in the following format:

Prediction: [Real or Fake or NEI(Not Enough Information)]
Explanation: [Provide a detailed explanation of the factors that led to your conclusion. Consider elements such as image quality, inconsistencies, artifacts, lighting, context, and any other relevant indicators that influenced your decision.]
"""

ICL1_AUTHEN_PREDICTION = """You are an advanced AI model trained to assess images for authenticity. Your task is to analyze the given image and determine whether it is real or fake. Provide a clear and concise response with your prediction and an explanation supporting your decision.

Please output your response in the following format:

Prediction: [Real or Fake or NEI(Not Enough Information)]
Explanation: [Provide a detailed explanation of the factors that led to your conclusion. Consider elements such as image quality, inconsistencies, artifacts, lighting, context, and any other relevant indicators that influenced your decision.]


For the first image:

Prediction: {}
Explanation: {}

For the second image:
"""

ICL2_AUTHEN_PREDICTION = """You are an advanced AI model trained to assess images for authenticity. Your task is to analyze the given image and determine whether it is real or fake. Provide a clear and concise response with your prediction and an explanation supporting your decision.

Please output your response in the following format:

Prediction: [Real or Fake or NEI(Not Enough Information)]
Explanation: [Provide a detailed explanation of the factors that led to your conclusion. Consider elements such as image quality, inconsistencies, artifacts, lighting, context, and any other relevant indicators that influenced your decision.]


For the first image:
Prediction: {}
Explanation: {}

For the second image:
Prediction: {}
Explanation: {}

For the third image:
"""



AUTHEN_PROMPTS = {
    'cot': COT_AUTHEN_PREDICTION,
    'icl1': ICL1_AUTHEN_PREDICTION,
    'icl2': ICL2_AUTHEN_PREDICTION,
}




