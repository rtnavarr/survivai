---
layout: default
title: Final Report
---

## Video Summary of Project
<iframe width="560" height="315" src="https://www.youtube.com/embed/_RrEJiJDdLg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
** Replace this with final video

## Project Summary
Wood is the first item that any new player to Minecraft must learn to collect, and it is always a useful resource to have in your inventory. Our agent’s task is to automate the first and most essential step that a Minecraft player takes to survive: identify and harvest wood in order to craft tools that are ultimately used to upgrade equipment and progress in “survival” mode Minecraft. Computer vision plays a key role in Surviv.AI; our agent takes in a colormap view of RGB and depth values, detects the objects it needs to collect in the frame, and navigates to these objects. The goal is to have the agent rely solely on vision, rather than ObservationFromGrid(as used in assignment 2) to complete simple tasks such as gathering wood.

<img src="http://www.minecraft101.net/guides/images/first-night/03-getting-wood.jpg"/>

## Approach


## Evaluation


## Resources Used
Surviv.ai was built using the following resources:<br>
- <a href="https://www.microsoft.com/en-us/research/project/project-malmo/">Microsoft's Project Malmo</a><br>
- <a href="https://microsoft.github.io/malmo/0.30.0/Schemas/Mission.html#element_AgentHandlers">Malmo XML Schema Documentation</a><br>
- <a href="https://github.com/kchian/ForkThePork">ForkThePork project from Fall 2020</a><br>
- <a href="https://github.com/microsoft/malmo/blob/master/Malmo/samples/Python_examples/radar_test.py">Malmo radar_test.py Tutorial</a><br>
- <a href="https://github.com/microsoft/malmo/blob/master/Malmo/samples/Python_examples/depth_map_runner.py">Malmo depth_runner.py Tutorial</a><br>
- <a href="http://microsoft.github.io/malmo/0.14.0/Python_Examples/Tutorial.pdf">Malmo tutorial_5.py</a><br>
- <a href="https://github.com/microsoft/malmo/blob/master/Schemas/Types.xsd">Malmo types.xsd</a><br>
- <a href="https://openai.com/blog/openai-baselines-ppo/">OpenAI Documentation on PPO</a><br>
- <a href="https://medium.com/datadriveninvestor/which-reinforcement-learning-rl-algorithm-to-use-where-when-and-in-what-scenario-e3e7617fb0b1#:~:text=It%20can%20be%20observed%20that,hence%20requires%20several%20add%2Dons.&text=TD3%20and%20TRPO%20work%20well,lack%20the%20faster%20convergence%20rate">Which RL Algorithm to use- where, when, and in what scenario? - Medium</a><br>
- <a href="https://medium.com/intro-to-artificial-intelligence/proximal-policy-optimization-ppo-a-policy-based-reinforcement-learning-algorithm-3cf126a7562d#:~:text=Proximal%20Policy%20Optimization(PPO)%2D,732%20Followers">Intro to PPO - Medium</a><br>
- <a href="https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-3-4-82081ea58146">Coding PPO - Medium</a><br>
- <a href="https://www.youtube.com/watch?v=5P7I-xPq8u8">Policy Gradient Method and PPO: Diving into Deep RL - Youtube</a><br>
- <a href="https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/">Convolutional Neural Networks Tutorial in PyTorch</a><br>
- <a href="https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53">A Comprehensive Guide to Convolutional Neural Networks</a><br>
- <a href="http://www.minecraft101.net/g/first-night.html">Survival in Minecraft (source of image in project summary)</a><br>
