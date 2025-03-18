# Impulse Guard

## What It Does
Stress Manager uses voice and facial recognition to detect whether you might be stressed while making a decision. If stress is detected, the app warns you to reconsider your choice before proceeding.
It uses a VGG16 model trained on the KDEF dataset, which contains facial expressions labeled by emotion. This allows us to classify a person’s facial expression based on their emotional state.
It leverages Google Vision API to recognize faces and crop images to focus on the face. The cropped image is then passed into our trained VGG16 model to detect negative emotions.
Additionally, it uses voice stress analysis model to determine if the user is stressed based on their speech patterns. By combining insights from both the facial and voice analysis models, Stress Manager assesses whether the user is in a suitable mental state to make a decision.
