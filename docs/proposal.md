---
layout: default
title: Proposal
---

## Meeting Time
3:30 - 5:00PM on Tuesdays

---

## Summary of the Project
The goal of this project is to create an agent that can “survive”. 

In a normal Minecraft survival game, the player focuses on acquiring/crafting a set of items needed to progress through the game. These first few items would be wood, sticks, crafting bench, wooden pickaxe, cobblestone, coal ore, etc. The agent, in this case, would be taking in a view (Computer Vision) of the blocks in front of them as input to locate the right blocks needed to collect/craft the desired items. An action (movement, mining, crafting, etc.) would be executed as output upon evaluation of the input state. The goal is to gather all the items necessary for “survival” within a reasonable time frame.

Applications for this project are rather broad in the real world, as it pertains to object collection and utilization. An example might be making a peanut-butter jelly sandwich. Items needed for this task would be a knife, plate, bread, peanut butter, jelly, etc. 

---

## AI/ML Algorithms
We plan on associating each “task” with a reward value and using reinforcement learning to teach the agent the optimal steps to progress in Minecraft. Along the way, we might need to implement object detection so the agent can recognize objects outside of a 1-dimensional grid.

---

## Evaluation Plan
We plan to evaluate our agent’s performance by examining the time it takes for the agent to recognize and pick up wood. Our baseline is that the agent will be able to successfully find wood in a basic training world within a minute; this world would be completely flat and include wood located within X number of blocks of the agent. Our sanity cases would ensure that the agent is able to identify wood logs or trees using image classification, navigate to these objects successfully, and collect the correct quantity before the mission timer runs out. Beyond our baseline, we would also train our agent to collect wood within a shorter time frame. Additionally, our moonshot case would be for the agent to navigate to and collect other items needed for more complex recipes.

---

## Projected Completion by Status Report
We plan to have the following functionalities completed:
- Generation of a “training map” (potentially random, feasible maps)
- RL agent that detects and moves toward a wood log or tree
- Reward system based on time taken since start

---

## Appointment with the Instructor
We scheduled an appointment to meet with Professor Singh at 02:30pm PDT on Thursday, January 21, 2021.