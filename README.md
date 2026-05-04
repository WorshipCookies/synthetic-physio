# Physiological Synthetic Generator 
This generator was created for teaching purposes. 

It consists of 3 modalities (EDA, ECG and Facial AU) that are generated allowing students to process, analyse and test during laboratorial type lectures. 

## Simulated Environment
The data simulates 15 minute sessions of gameplay of a specific horror game, where 2 "jump scare" triggers happen during the early and late game; whilst in-between the game is slowly increasing the enviromental tension. 

## Simulated Noise and Artefacts
The data attemtpts to simulate noise and artefacts that happen over the course of the session, so students are able to deal with these concepts actively during their analysis. 

Each playthrough is also "simulated" in a sense that the events aren't synchronized between each playthrough. This simulates the concept that players play through the game at difference paces, and students must be able to adapt to these affordances.

Lastly, each modality is "recorded" at different sampling rates:

- EDA: 4 Hz
- ECG: 100 Hz
- Facial AU's: 30 Hz 

This forces the students to understand that different modalities can also be limited based on the sampling rate and must be able to deal with these conditions also. 