# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea

# Unified VQA template — aligned with galaxea_cot_processor chat format.
# Placeholders <chat_user_prefix>, <chat_user_suffix>, <chat_assistant_prefix>, <eos>
# are resolved by SpecialTokenManager.resolve_template() at runtime per model_type.
VQA_TEMPLATE = """<chat_user_prefix><image_image_!><question_text_!><chat_user_suffix><chat_assistant_prefix><EOC><answer_text><eos>"""

# Legacy alias — VQA datasets import this name
PALIGEMMA_VQA_TEMPLATE = VQA_TEMPLATE
