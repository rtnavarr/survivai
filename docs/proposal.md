---
layout: default
title: Proposal
---

## Summary of the Project
The goal of this project is to create an agent that can “survive”. In a normal Minecraft survival game, the player focuses on acquiring/crafting a set of items needed to progress through the game. These first few items would be wood, sticks, crafting bench, wooden pickaxe, cobblestone, coal ore, etc. The agent, in this case, would be taking in a view (Computer Vision) of the blocks in front of them as input to locate the right blocks needed to collect/craft the desired items. An action (movement, mining, crafting, etc.) would be executed as output upon evaluation of the input state. The goal is to gather all the items necessary for “survival” within a reasonable time frame.
Applications for this project are rather broad in the real world, as it pertains to object collection and utilization. An example might be making a peanut-butter jelly sandwich. Items needed for this task would be a knife, plate, bread, peanut butter, jelly, etc. 

---

## AI/ML Algorithms
We plan on associating each “task” with a reward value and using reinforcement learning to teach the agent the optimal steps to progress in Minecraft. Along the way, we might need to implement object detection so the agent can recognize objects outside of a 1-dimensional grid.

---

## Evaluation Plan
We plan to evaluate our agent’s performance using metrics such as the cumulative score of rewards from each task completed, the time it took to complete each task, and the remaining health score that the agent has by the end of the mission. Our baseline is that the agent will be able to successfully ‘complete’ a feasible survival world used for training; this world would be completely flat and include all the items that the agent needs to collect for survival, such as trees, stones, and ores, all located within X number of blocks of the agent. Our sanity cases would ensure that the agent is able to recognize the objects it needs to collect from the world(using computer vision), navigate to these objects successfully, and collect the quantity that is required for recipes. Beyond the baseline, we would also train the agent to perform on multiple other worlds with geographic obstacles such as uneven terrain and trees between the agent and the items it needs to collect. Finally, our moonshot case would be for the agent to navigate and  ‘complete’ survival on a random world created by the default WorldGenerator that it has not seen before.
