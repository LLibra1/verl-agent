# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --------------------- ALFWorld --------------------- #

ALFWORLD_TEMPLATE = """
You will be presented with a situation where you need to choose between multiple possible actions. Your task is to analyze the situation and provide reasoning about why the expert action is preferable to the alternative action, based on the differences between their resulting states.  
• Situation Description (si): {Situation Description}  
• Expert Action (ai): {Expert Action}  
• Resulting State of Expert Action (si+1): {Future State of Expert Action}  
• Alternative Actions: 
  1. Action a1  i : {Alt Action 1}, resulting state s1  i : {State 1} 
  2. Action a2  i : {Alt Action 2}, resulting state s2  i : {State 2} 
  3. . . .  
Provide a detailed self-reflection as an internal monologue that demonstrates your reasoning process for the current situation. Your monologue should: 
  1. Analyze the current situation and the goal. 
  2. Compare the resulting states: examine what state the expert action leads to (si+1) versus what state the alternative action leads to (s1  i), and identify the key differences between these states. 
  3. Based on these state differences, explain why the alternative action is less optimal — highlight potential limitations or inefficiencies revealed by its resulting state. 
  4. Justify why the expert action is most suitable, grounded in the actual state transitions observed.  
Guidelines:  
• Stay strictly within the provided information. 
• Avoid meta-commentary about being an AI. 
• Use natural, step-by-step reasoning. 
• Focus on logical decision-making grounded in the observed state differences. 
Output: Directly write the self-reflection monologue, no extra headings, disclaimers, or external notes.
"""