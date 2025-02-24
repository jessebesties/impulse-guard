# Impulse Guard
[DevPost](https://devpost.com/software/stress-6m2pl9)

## Inspiration
Have you ever made a rash decision and later realized it was the wrong choice? That’s why we present to you Stress Manager—an application designed to help you make more thoughtful decisions by analyzing your stress levels.

## What It Does
Stress Manager uses voice and facial recognition to detect whether you might be stressed while making a decision. If stress is detected, the app warns you to reconsider your choice before proceeding.

## How We Built It
We used a VGG16 model trained on the KDEF dataset, which contains facial expressions labeled by emotion. This allows us to classify a person’s facial expression based on their emotional state.

We also used Google Vision API to recognize faces and crop images to focus on the face. The cropped image is then passed into our trained VGG16 model to detect negative emotions.

Additionally, we implemented a voice stress analysis model to determine if the user is stressed based on their speech patterns. By combining insights from both the facial and voice analysis models, Stress Manager assesses whether the user is in a suitable mental state to make a decision.

## Challenges We Ran Into
One major challenge was training the VGG16 model. The original dataset contained only 5,000 images, and 30% were unusable due to side profiles. This caused our model to predict only one dominant emotion, significantly reducing accuracy. To address this, we augmented the dataset by flipping, adjusting brightness, rotating, and shifting images. This significantly improved the model’s performance.

Since Georgia Tech’s GPUs were down, we had to train our model using Google Colab’s GPUs instead.

Another challenge was working with audio processing. The MediaRecorder API in JavaScript produced a data format that was incompatible with our model. Additionally, transferring audio from the front end to the backend was problematic. We resolved these issues through extensive debugging and by using FFmpeg to convert audio from WebP to WAV format.

## Accomplishments That We're Proud Of
We’re proud that our app effectively determines whether a user is in the right state of mind to make a decision—whether it’s buying stocks or purchasing something on Amazon. While the model's performance can still be improved with more training and a diverse dataset, we believe Stress Manager has the potential to help many people make fewer impulsive mistakes.

## What We Learned
We learned a lot about training deep learning models, data augmentation, working with APIs, handling real-time audio and image processing, and so much more!

## What's Next for Stress Manager
We plan to improve our model’s accuracy by training on a larger and more diverse dataset. Additionally, we aim to enhance real-time processing, refine our user interface, and explore potential mobile and browser extensions for wider accessibility.
