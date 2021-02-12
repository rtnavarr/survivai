---
layout: default
title:  Status
---
## Video Summary of Project

## Project Summary
Wood is essential for crafting objects in Minecraft, and in general, it can be a useful item to have in one’s inventory. Our project focuses on helping the agent identify and harvest wood, which would ultimately enable it to craft tools that could then be used to collect other key items in a “survival” mode game of Minecraft. Computer vision with the use of a color map plays an essential role in our project; the agent takes in a view, detects the objects it needs to collect(ie, forms of wood such as simple logs and more complex trees) in the frame, and navigates to these objects. The goal is to have the agent rely solely on vision, rather than observation from grid-- which consitutes "cheating"-- to complete simple tasks such as harvesting wood. In later stages, we hope to implement a timer system that dictates the timeframe in which the agent must collect a certain quantity of wood, or perhaps a crafting goal such as building a table from the objects collected.

## Approach

## Evaluation

## Remaining Goals and Challenges
Our prototype is limited in that it is not yet capable of pitching/tilting its perspective to recognize that there are wood blocks below(y=1) and above(y=3) the ones placed at y=2. We believe that this is primarily due to the fact that our agent only examines a single pixel at the center of the current colormap frame, rather than looking at a slice of multiple pixels, to compare the current view's RGB values to those expected for wood. In the coming weeks, we thus plan to make our agent's wood collection more "complete" by modifying our detection algorithm to encapsulate a broader view of our target logs. We hope to achieve this by enlarging the RGB comparison window and have the agent continue to attack as long as there are remaining pixels with the wood RGB value that fall within that window. Instead of evaluating the RGB value of a single center pixel and using this as the only criteria for halting a turn and subsequently attacking, we plan to construct a vertical, horizontal, or cross-shaped window that examines a region of pixels and gets the majority "vote" color to compare to the expected color for wood. Another major goal of ours is to modify the agent to recognize and collect wood when it is presented in more complex forms such as trees, as opposed to simple logs. We also anticipate adding complexity to our world by introducing obstacles such as lava, or other non-wood items like stone blocks that the agent needs to locate and harvest in addition to wood.

Moreover, since the transition probabilities are unknown, the action space is continuous, and the state space is very large, we opted to use a deep reinforcement learning approach with PPO. However, in future iterations, we also plan to experiment with other algorithms such as Deep Q-learning, A3C, or TRPO, and visually compare the results with those from our current iteration to determine if there is a better approach than PPO. Additionally, our agent currently turns at a very slow rate of 0.05. We set this low value to ensure that as the agent was training, it was able to have enough time to register the updated RGB values with the arrival of each new frame as the agent moved around. For our next iteration, we plan to find a way to enhance our agents performance by making it move faster(and thus locate and navigate to the wood blocks more quickly), and tune this parameter to find the optimal turning speed.

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