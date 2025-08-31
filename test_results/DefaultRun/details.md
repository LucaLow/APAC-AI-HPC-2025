# Job Description
This was just a regular test to determine a baseline for running deepseek across 16 Gpu's on 2 nodes.

# Findings:
- There appears to be a whole load of errors at the end of the program, further investigation reveals these are most likely related to node 0 quitting smoothly and sending EOL messages to node 1, node 1 doesn't take that very well, hangs or something, node 0 doesn't give a fuck and finishes, and then the time limit is reached as node 1 continues to hang, resulting in the running over time errors.